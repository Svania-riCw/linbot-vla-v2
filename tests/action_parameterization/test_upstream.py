"""Execute selected ORIGINAL upstream bodies with minimal dependency boundaries.

NumPy tests exercise actual joint apply/unapply and the actual non-clipping
Normalizer, with feature mapping bypassed explicitly. Torch tests additionally
exercise the original quaternion module. No VLA import/weights are required.
"""
import ast
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from numpy.testing import assert_allclose
from action_contracts import encode_ee_delta

ROOT = Path(os.environ.get('LINGBOT_UPSTREAM', str(Path(__file__).resolve().parents[2])))
DATA = ROOT / 'lingbotvla/data/vla_data'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def extract(path, class_name=None, method_names=None, top_names=None):
    """Compile complete selected definitions, preserving their original bodies."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    if class_name:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
        if len(nodes) != len(method_names):
            raise AssertionError('upstream method missing')
    else:
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in top_names]
        if len(nodes) != len(top_names):
            raise AssertionError('upstream definition missing')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    namespace = {'np': np}
    if HAS_TORCH:
        import torch
        namespace['torch'] = torch
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


class OriginalJointTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA.is_dir():
            raise FileNotFoundError(f'set LINGBOT_UPSTREAM to pinned source checkout: {ROOT}')
        cls.norm_ns = extract(DATA / 'transform.py', top_names={'dict_apply', 'Normalizer'})
        cls.methods = extract(DATA / 'utils.py', class_name='FeatureTransform', method_names={'apply', 'unapply'})
        cls.methods['_is_quaternion_relative_type'] = extract(
            DATA / 'ee_pose_transform.py', top_names={'_is_quaternion_relative_type'}
        )['_is_quaternion_relative_type']

    def make_transform(self, subtract=True):
        # These distinct action/state stats expose order mistakes: subtract must
        # precede normalize, and unnormalize must precede adding the anchor.
        stats = {
            'action.arm.position': {'q01': np.full((4, 12), -2.), 'q99': np.full((4, 12), 3.)},
            'observation.state.arm.position': {'q01': np.full(12, -10.), 'q99': np.full(12, 10.)},
        }
        normalizer = self.norm_ns['Normalizer'](stats, {key: 'bounds_99_woclip' for key in stats})
        obj = SimpleNamespace(
            org_features={'actions': ['action'], 'states': ['observation.state']},
            actions_convert_from_state={}, chunk_size=4,
            actions=['action.arm.position', 'action.effector.position'],
            action_subtract_state={'action.arm.position': subtract, 'action.effector.position': False},
            action_relative_type={}, normalizer=normalizer,
            return_item_befor_padding=True,
            convert_features=lambda item, w_action: item, reverse_features=lambda item: item,
        )
        item = {
            'action.arm.position': np.arange(48, dtype=np.float32).reshape(4, 12)/10,
            'observation.state.arm.position': np.arange(12, dtype=np.float32)/5,
            'action.effector.position': np.array([[0., 1.], [1., 0.], [.2, .8], [.9, .1]]),
            'observation.state.effector.position': np.array([.3, .7]),
            'action_is_pad': np.zeros(4, dtype=bool),
        }
        return obj, item

    def test_original_joint_delta_normalize_decode_order(self):
        obj, item = self.make_transform()
        original = copy.deepcopy(item)
        encoded = self.methods['apply'](obj, item)
        q = original['action.arm.position'] - original['observation.state.arm.position']
        expected = (q+2)/(5+1e-6)*2-1
        assert_allclose(encoded['action.arm.position'], expected, atol=1e-6)
        assert_allclose(encoded['action.effector.position'], original['action.effector.position'])
        decoded = self.methods['unapply'](obj, encoded)
        assert_allclose(decoded['action.arm.position'], original['action.arm.position'], atol=1e-6)
        assert_allclose(decoded['observation.state.arm.position'], original['observation.state.arm.position'], atol=1e-6)

    def test_original_absolute_path_unchanged(self):
        obj, item = self.make_transform(subtract=False)
        original = copy.deepcopy(item)
        decoded = self.methods['unapply'](obj, self.methods['apply'](obj, item))
        assert_allclose(decoded['action.arm.position'], original['action.arm.position'], atol=1e-6)

    def test_original_missing_anchor_raises(self):
        obj, item = self.make_transform()
        del item['observation.state.arm.position']
        with self.assertRaises(ValueError):
            self.methods['apply'](obj, item)

    def test_original_norm_horizon_mismatch_rejected(self):
        obj, item = self.make_transform()
        item['action.arm.position'] = np.zeros((5, 12))
        with self.assertRaisesRegex(ValueError, 'horizon mismatch'):
            self.methods['apply'](obj, item)

    def test_original_woclip_extreme_values_roundtrip(self):
        obj, item = self.make_transform()
        item['action.arm.position'] *= 100
        original = copy.deepcopy(item)
        encoded = self.methods['apply'](obj, item)
        self.assertGreater(encoded['action.arm.position'].max(), 1.5)
        decoded = self.methods['unapply'](obj, encoded)
        assert_allclose(decoded['action.arm.position'], original['action.arm.position'], atol=1e-4)

    @unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable: original clipping path NOT tested')
    def test_original_clipping_is_not_invertible(self):
        import torch
        norm = self.norm_ns['Normalizer']({'x': {'q01': 0., 'q99': 1.}}, {'x': 'bounds_99'})
        original = {'x': torch.tensor([0.5, 10.])}
        encoded = norm.normalize(original)
        decoded = norm.unnormalize(encoded)
        self.assertAlmostEqual(decoded['x'][0].item(), .5, places=5)
        self.assertLess(decoded['x'][1].item(), 2.)
        self.assertEqual(encoded['x'][1].item(), 1.5)


@unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable: original EE module NOT tested')
class OriginalEETests(unittest.TestCase):
    def test_original_local_ee_matches_independent_scipy_oracle_and_gradients(self):
        import torch
        spec = importlib.util.spec_from_file_location('original_ee', DATA / 'ee_pose_transform.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(55)
        state = rng.normal(size=(3, 16)).astype('float32')
        actions = rng.normal(size=(3, 5, 16)).astype('float32')
        expected = encode_ee_delta(actions, state)
        indices = list(range(7)) + list(range(8, 15))
        a = torch.tensor(actions[..., indices], requires_grad=True)
        s = torch.tensor(state[:, None, indices])
        result = module.relative_pose_quaternion(a, s, relative_type='quaternion_local')
        assert_allclose(result.detach().numpy(), expected[..., indices], atol=2e-6)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(a.grad).all().item())
        self.assertGreater(a.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
