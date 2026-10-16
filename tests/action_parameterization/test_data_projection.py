"""Exercise actual loader methods without unrelated image decoding."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from test_upstream import extract

ROOT=Path(__file__).resolve().parents[2]


class ProjectionTests(unittest.TestCase):
    def test_statistics_disable_embedded_image_columns_but_training_keeps_them(self):
        source=ast.parse((ROOT/'lingbotvla/data/vla_data/base_dataset.py').read_text(encoding='utf-8'))
        node=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='LeRobotDataset')
        node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__']
        class HF:
            column_names=['action','observation.state','camera','video']
            def remove_columns(self,keys):
                result=HF();result.column_names=[key for key in self.column_names if key not in keys];return result
        original=HF()
        class Base:
            def __init__(self,repo_id,**kwargs):
                self.meta=SimpleNamespace(camera_keys=['camera','video','missing'])
                self.hf_dataset=original
        scope=dict(BaseLeRobotDataset=Base)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_loader_init','exec'),scope)
        loader=scope['LeRobotDataset']
        self.assertEqual(loader('test',load_image=False).hf_dataset.column_names,['action','observation.state'])
        self.assertIs(loader('test',load_image=True).hf_dataset,original)

    def test_future_queries_project_columns_preserving_indices_and_video_skip(self):
        methods=extract(ROOT/'lingbotvla/data/vla_data/base_dataset.py',class_name='LeRobotDataset',
                        method_names={'_query_hf_dataset'})
        calls=[]
        class HF:
            def select_columns(self,keys):
                calls.append(keys)
                class Selected:
                    def __getitem__(self,indices):
                        return {keys[0]:[torch.tensor([float(index)]) for index in indices]}
                return Selected()
            def __getitem__(self,indices):
                raise AssertionError('Unprojected row lookup would decode unrelated images')
        obj=SimpleNamespace(hf_dataset=HF(),meta=SimpleNamespace(video_keys=['video']),
                            _absolute_to_relative_idx={10:2,11:3})
        # Extracted method resolves helpers in its original source namespace.
        method=methods['_query_hf_dataset']
        method.__globals__['_to_relative_indices']=lambda dataset,indices:[dataset._absolute_to_relative_idx[i] for i in indices]
        result=method(obj,{'action':[10,11,11],'observation.images.future':[11,10],'video':[10]})
        self.assertEqual(calls,[['action'],['observation.images.future']])
        torch.testing.assert_close(result['action'],torch.tensor([[2.],[3.],[3.]]))
        torch.testing.assert_close(result['observation.images.future'],torch.tensor([[3.],[2.]]))
        self.assertNotIn('video',result)


if __name__=='__main__':unittest.main()
