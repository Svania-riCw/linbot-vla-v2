import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation
from action_contracts import (
    ARM, GRIP, encode_joint_delta, decode_joint_delta, encode_ee_delta,
    decode_ee_delta, reorder_ee_quaternions, weighted_flow_l1, recover_clean_action,
)
from check_dataset import check_arrays


class JointTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.state = self.rng.normal(size=(3, 14)).astype('float32')
        self.actions = self.rng.normal(size=(3, 50, 14)).astype('float32')

    def test_roundtrip_float32_and_no_mutation(self):
        original, state = self.actions.copy(), self.state.copy()
        d = encode_joint_delta(self.actions, self.state)
        assert_allclose(decode_joint_delta(d, self.state), original, atol=1e-6)
        assert_array_equal(self.actions, original)
        assert_array_equal(self.state, state)

    def test_grippers_absolute_and_arms_in_original_order(self):
        s = np.arange(14, dtype=float)
        a = np.stack([s + 1, s + 2])
        d = encode_joint_delta(a, s)
        assert_allclose(d[:, ARM], np.array([np.ones(12), np.full(12, 2)]))
        assert_array_equal(d[:, GRIP], a[:, GRIP])

    def test_frozen_anchor_not_stepwise_or_cumulative(self):
        s = np.zeros(14)
        a = np.ones((3, 14)) * np.array([1., 3., 6.])[:, None]
        d = encode_joint_delta(a, s)
        assert_array_equal(d[:, 0], [1., 3., 6.])
        self.assertFalse(np.allclose(np.cumsum(d[:, 0]), a[:, 0]))

    def test_constant_target_and_zero_motion(self):
        a = np.repeat(self.state[:, None, :], 50, axis=1)
        d = encode_joint_delta(a, self.state)
        assert_array_equal(d[..., ARM], 0)
        assert_array_equal(d[..., GRIP], a[..., GRIP])

    def test_invalid_state_shape_nan_and_integer_rejected(self):
        for a, s in [(self.actions, self.actions), (self.actions.astype(int), self.state),
                     (self.actions * np.nan, self.state)]:
            with self.assertRaises(ValueError):
                encode_joint_delta(a, s)


class EETests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.state = rng.normal(size=(2, 16))
        self.actions = rng.normal(size=(2, 8, 16))
        for start in (3, 11):
            self.state[:, start:start+4] = Rotation.random(2, random_state=rng).as_quat()
            self.actions[..., start:start+4] = Rotation.random(16, random_state=rng).as_quat().reshape(2, 8, 4)

    def test_se3_roundtrip_and_grippers(self):
        d = encode_ee_delta(self.actions, self.state)
        out = decode_ee_delta(d, self.state)
        for start in (0, 8):
            assert_allclose(out[..., start:start+3], self.actions[..., start:start+3], atol=1e-12)
            r1 = Rotation.from_quat(out[..., start+3:start+7].reshape(-1, 4))
            r2 = Rotation.from_quat(self.actions[..., start+3:start+7].reshape(-1, 4))
            self.assertLess((r1.inv()*r2).magnitude().max(), 1e-12)
        assert_array_equal(d[..., [7, 15]], self.actions[..., [7, 15]])

    def test_local_frame_known_90_degree_rotation(self):
        state = np.zeros(16)
        state[3:7] = Rotation.from_euler('z', 90, degrees=True).as_quat()
        state[11:15] = [0, 0, 0, 1]
        actions = state[None].copy()
        actions[0, :3] = [0, 1, 0]
        assert_allclose(encode_ee_delta(actions, state)[0, :3], [1, 0, 0], atol=1e-12)

    def test_quaternion_sign_equivalence(self):
        flipped = self.actions.copy()
        flipped[..., 3:7] *= -1
        flipped[..., 11:15] *= -1
        assert_allclose(encode_ee_delta(flipped, self.state), encode_ee_delta(self.actions, self.state), atol=1e-12)

    def test_identity_delta_is_identity_quaternion_not_zero_vector(self):
        d = encode_ee_delta(self.state[:, None], self.state)
        for start in (0, 8):
            assert_allclose(d[..., start:start+7], np.broadcast_to([0, 0, 0, 0, 0, 0, 1], (2, 1, 7)), atol=1e-12)

    def test_order_conversion_known_quaternion(self):
        a = np.zeros((1, 16))
        a[:, 3:7] = [1, 0, 0, 0]  # identity in wxyz
        a[:, 11:15] = [0, 1, 0, 0]  # pi about X in wxyz
        b = reorder_ee_quaternions(a, 'wxyz', 'xyzw')
        assert_array_equal(b[:, 3:7], [[0, 0, 0, 1]])
        assert_array_equal(b[:, 11:15], [[1, 0, 0, 0]])
        assert_array_equal(reorder_ee_quaternions(b, 'xyzw', 'wxyz'), a)

    def test_zero_quaternion_rejected(self):
        self.actions[..., 3:7] = 0
        with self.assertRaises(ValueError):
            encode_ee_delta(self.actions, self.state)


class FlowTests(unittest.TestCase):
    def test_clean_action_identity_at_zero_middle_one(self):
        rng = np.random.default_rng(12)
        a, n = rng.normal(size=(2, 3, 50, 14))
        s = np.array([0., .37, 1.])
        x = s[:, None, None]*n + (1-s[:, None, None])*a
        assert_allclose(recover_clean_action(x, n-a, s), a, atol=1e-12)

    def test_unit_weights_match_global_baseline_with_unequal_mask_counts(self):
        p = np.arange(24, dtype=float).reshape(2, 3, 4)
        m = np.ones(p.shape, dtype=bool)
        m[0, :, 1:] = False
        expected = (np.abs(p)*m).sum()/m.sum()
        self.assertEqual(weighted_flow_l1(p, p*0, m), expected)

    def test_padding_channels_dont_affect_loss(self):
        p = np.ones((1, 2, 55))
        m = np.zeros(p.shape, dtype=bool)
        m[..., :14] = True
        p[..., 14:] = 1e8
        self.assertEqual(weighted_flow_l1(p, p*0, m), 1.)

    def test_hand_computed_weights_and_scale_invariance(self):
        p = np.array([[[1., 2.], [3., 4.]]])
        m = np.ones(p.shape, dtype=bool)
        self.assertAlmostEqual(weighted_flow_l1(p, p*0, m, [1, 3], [2, 1]), 29/12)
        self.assertAlmostEqual(weighted_flow_l1(p, p*0, m, [10, 30], [2, 1]), 29/12)

    def test_empty_negative_nan_weights_rejected(self):
        p = np.ones((1, 2, 2))
        for weights in ([0, 0], [-1, 2], [np.nan, 1]):
            with self.assertRaises(ValueError):
                weighted_flow_l1(p, p*0, np.ones(p.shape, dtype=bool), weights)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(observed_state=np.ones((2, 14)), actions=np.ones((2, 3, 14))*2,
                         action_is_pad=np.array([[False, False, True], [False, False, False]]),
                         episode_id=np.array([8, 9]), action_episode_id=np.array([[8, 8, 8], [9, 9, 9]]))

    def test_valid_export_and_pad_excluded_from_statistics(self):
        self.data['actions'][0, 2] = 100
        result = check_arrays(self.data)
        self.assertEqual(result['valid_targets'], 5)
        assert_array_equal(result['absolute_joint_abs_p95'], np.full(12, 2.))

    def test_cross_episode_rejected(self):
        self.data['action_episode_id'][0, 1] = 9
        with self.assertRaisesRegex(ValueError, 'boundary'):
            check_arrays(self.data)

    def test_all_padding_or_interior_padding_rejected(self):
        for row in ([True, True, True], [False, True, False]):
            self.data['action_is_pad'][0] = row
            with self.assertRaises(ValueError):
                check_arrays(self.data)


if __name__ == '__main__':
    unittest.main()
