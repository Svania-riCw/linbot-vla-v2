import ast
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import dataclass, field
from typing import Optional

from ysj_utils.resume_joint_delta import continuation_config, build_commands, inspect_checkpoint, select_checkpoint, validate_source_experiment, ROOT

@dataclass
class SavedOptions:
    basic_modules: list[str] = field(default_factory=list)
    optional_steps: Optional[int] = None
    labels: Optional[list[str]] = None

@dataclass
class SavedRoot:
    train: SavedOptions = field(default_factory=SavedOptions)


class ResumeTests(unittest.TestCase):
    def test_spline_resume_preserves_representation_precision_and_padding(self):
        train = dict(max_steps=5000, global_batch_size=128, micro_batch_size=16, spline_control_points=16,
                     spline_degree=3, mask_action_padding=False, enable_fp32=True, enable_mixed_precision=True,
                     joint_ee_aux_weight=-1., flow_gibbs_epsilon=0., flow_dimension_weights=None, flow_horizon_weights=None)
        validate_source_experiment(train, 'joint_absolute', 'spline_absolute')
        original = dict(train=train, data=dict(prefetch_factor=1, norm_stats_file='original'))
        continued = continuation_config(original, Path('/continued'), '/original/global_step_5000', 10000, 1)
        for field in ('spline_control_points','spline_degree','mask_action_padding','enable_fp32','enable_mixed_precision'):
            self.assertEqual(continued['train'][field], train[field])
        self.assertEqual(continued['data'], original['data'])
        self.assertEqual(continued['train']['lr_schedule_steps'], 5000)
        self.assertTrue(continued['train']['require_resume'])
        for field, bad in [('spline_control_points',8),('mask_action_padding',True),('enable_fp32',False),('flow_gibbs_epsilon',.5)]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_source_experiment(dict(train, **{field:bad}), 'joint_absolute', 'spline_absolute')
        with self.assertRaises(ValueError): validate_source_experiment(train,'joint_delta','spline_absolute')

    def test_spline_wrapper_and_uncompiled_eval_target_global_10000(self):
        from ysj_utils.resume_spline import command_defaults
        defaults = command_defaults()
        self.assertEqual(defaults[defaults.index('--inference-compile')+1], 'false')
        self.assertEqual(defaults[defaults.index('--additional-steps')+1], '5000')
        _, command = build_commands(Path('/cfg'),Path('/out'),[str(i) for i in range(8)],10000,use_compile=False)
        self.assertEqual(command[command.index('--use-compile')+1], 'false')
        self.assertEqual(Path(command[command.index('--checkpoint')+1]).parts[-2:], ('global_step_10000','hf_ckpt'))
        self.assertTrue(command[command.index('--summary-output')+1].endswith('evaluation_compile_false.json'))

    def test_resolved_yaml_empty_list_and_null_round_trip(self):
        import argparse,dataclasses,typing,types,inspect,os,sys,yaml
        from collections import defaultdict
        from enum import Enum
        tree=ast.parse((ROOT/'lingbotvla/utils/arguments.py').read_text())
        names={'parse_args','_string_to_bool','_make_choice_type_function','_convert_str_dict'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
        scope=dict(argparse=argparse,fields=dataclasses.fields,MISSING=dataclasses.MISSING,
                   get_type_hints=typing.get_type_hints,Union=typing.Union,Literal=typing.Literal,
                   Optional=typing.Optional,List=typing.List,Dict=typing.Dict,Any=typing.Any,
                   Callable=typing.Callable,T=object,types=types,isclass=inspect.isclass,
                   Enum=Enum,sys=sys,os=os,json=json,yaml=yaml,defaultdict=defaultdict)
        exec(compile(module,'actual_yaml_parser','exec'),scope)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'saved.yaml'
            p.write_text('train:\n  basic_modules: []\n  optional_steps: null\n  labels: null\n')
            with patch('sys.argv',['train.py',str(p)]):out=scope['parse_args'](SavedRoot)
            self.assertEqual(out.train.basic_modules,[])
            self.assertIsNone(out.train.optional_steps)
            with patch('sys.argv',['train.py',str(p),'--train.basic_modules','x']):
                out=scope['parse_args'](SavedRoot)
            self.assertEqual(out.train.basic_modules,['x'])

    def test_continuation_preserves_experimental_variables(self):
        original = dict(train=dict(max_steps=5000, enable_fp32=False, mask_action_padding=True,
                                   global_batch_size=128, use_compile=True),
                        data=dict(train_path='unchanged', norm_stats_file='unchanged'))
        before = copy.deepcopy(original)
        result = continuation_config(original, Path('/new'), '/old/global_step_5000', 10000)
        self.assertEqual(original, before)
        self.assertEqual(result['data'], original['data'])
        for name in ('enable_fp32','mask_action_padding','global_batch_size','use_compile'):
            self.assertEqual(result['train'][name], original['train'][name])
        self.assertEqual(result['train']['max_steps'], 10000)
        self.assertEqual(result['train']['lr_schedule_steps'], 5000)
        self.assertTrue(result['train']['require_resume'])
        reduced = continuation_config(original,Path('/new'),'/old/global_step_5000',10000,prefetch_factor=1)
        self.assertEqual(reduced['data'],dict(original['data'],prefetch_factor=1))

    def test_compiled_evaluation_is_explicit_and_recorded(self):
        train, evaluate = build_commands(Path('/cfg'), Path('/out'), ['0','1'], 10000)
        self.assertIn('--nproc-per-node=2', train)
        self.assertEqual(Path(evaluate[evaluate.index('--checkpoint')+1]),Path('/out/checkpoints/global_step_10000/hf_ckpt'))
        self.assertEqual(evaluate[evaluate.index('--use-compile')+1], 'true')
        from ysj_utils.evaluate import main
        output = io.StringIO()
        with patch('sys.argv', ['evaluate.py','--checkpoint','/fake','--gpus','0','--mode','full',
                                '--use-compile','true','--dry-run']), contextlib.redirect_stdout(output):
            main()
        report = json.loads(output.getvalue())
        self.assertTrue(report['use_compile'])
        self.assertEqual(report['total_episodes'], 350)

    def test_checkpoint_requires_every_saved_rank_and_untruncated_optimizer(self):
        import torch
        import torch.distributed.checkpoint as dcp
        with tempfile.TemporaryDirectory() as temporary:
            cp = Path(temporary)/'global_step_5000'
            for component in ('model','optimizer'):
                dcp.save({'tensor':torch.ones(3)}, checkpoint_id=cp/component)
            (cp/'extra_state').mkdir()
            state = dict(global_step=5000,lr_scheduler={'last_epoch':5000,'_last_lr':[5e-5]},
                         train_dataloader={},environ_meter={},torch_rng_state=torch.get_rng_state())
            for rank in range(2):torch.save(state,cp/'extra_state'/f'extra_state_rank_{rank}.pt')
            self.assertEqual(inspect_checkpoint(cp,2)['step'],5000)
            (cp/'extra_state/extra_state_rank_1.pt').unlink()
            with self.assertRaises(FileNotFoundError):inspect_checkpoint(cp,2)
            torch.save(state,cp/'extra_state/extra_state_rank_1.pt')
            shard=next((cp/'optimizer').glob('*.distcp'));shard.write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'truncated'):inspect_checkpoint(cp,2)

    def test_strict_optimizer_restore_does_not_swallow_errors(self):
        tree=ast.parse((ROOT/'lingbotvla/checkpoint/checkpointer.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DistributedCheckpointer')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='load')
        method.decorator_list=[]
        module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
        seen=[]
        def load(**kw):
            seen.append(kw)
            if len(seen)==2:raise RuntimeError('optimizer corrupt')
        scope=dict(Any=object,Dict=dict,os=__import__('os'),_MODEL_DIR='model',_OPTIMIZER_DIR='optimizer',
                   ModelState=lambda x:x,OptimizerState=lambda **kw:kw,
                   DefaultLoadPlanner=lambda **kw:kw,FileSystemReader=lambda p:p,
                   dcp=SimpleNamespace(load=load),logger=SimpleNamespace(info_rank0=lambda m:None))
        exec(compile(module,'actual_dcp_load','exec'),scope)
        state={'model':object(),'optimizer':object()}
        with self.assertRaisesRegex(RuntimeError,'optimizer corrupt'):
            scope['load'](None,'/checkpoint',state,strict_optimizer=True)
        self.assertFalse(seen[1]['planner']['allow_partial_load'])

    def test_scheduler_keeps_original_horizon_after_restore(self):
        import torch
        from lingbotvla.optim.lr_scheduler import build_lr_scheduler
        p=torch.nn.Parameter(torch.tensor(0.))
        optimizer=torch.optim.SGD([p],lr=1e-4)
        scheduler=build_lr_scheduler(optimizer,5000,lr=1e-4,lr_min=5e-5,lr_decay_style='cosine')
        scheduler.load_state_dict(dict(scheduler.state_dict(),last_epoch=5000,_last_lr=[5e-5]))
        for _ in range(3):optimizer.step();scheduler.step()
        self.assertEqual(optimizer.param_groups[0]['lr'],5e-5)

    def test_required_restore_rejects_missing_checkpoint_before_training(self):
        tree=ast.parse((ROOT/'tasks/vla/train_lingbotvla.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        guard=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test).startswith('args.train.require_resume and'))
        module=ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[]))
        scope={'args':SimpleNamespace(train=SimpleNamespace(require_resume=True,ckpt_manager='dcp',allow_partial_checkpoint=False)), 'candidates':[]}
        with self.assertRaisesRegex(RuntimeError,'existing complete DCP'):
            exec(compile(module,'actual_resume_guard','exec'),scope)

    def test_required_restore_rejects_load_error_without_fresh_fallback(self):
        tree=ast.parse((ROOT/'tasks/vla/train_lingbotvla.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        branch=next(n for n in main.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='candidates')
        module=ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[]))
        def broken(*a,**kw):raise OSError('corrupt DCP')
        scope=dict(args=SimpleNamespace(train=SimpleNamespace(require_resume=True,allow_partial_checkpoint=False)),
                   candidates=['/bad'],model=None,optimizer=None,Checkpointer=SimpleNamespace(load=broken),
                   logger=SimpleNamespace(info_rank0=lambda m:None))
        with self.assertRaisesRegex(RuntimeError,'Required training-state restore failed'):
            exec(compile(module,'actual_resume_branch','exec'),scope)


if __name__ == '__main__':
    unittest.main()
