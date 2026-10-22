"""d4rt probe targets under the input-resolution / observation-augmentation changes (CPU only).

* the canvas affine reproduces ObservationAugmenter's actual warp (384 view -> 224 canvas);
* with an affine, every target cell is the scene point under the augmented query pixel
  (round trip through the camera model on a synthetic episode);
* the 224 canvas resize and the PPU-side LeRobot directory names.
"""
import pathlib
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xsy_utils" / "spatial-exp"))

from d4rt_lingbot import data as D  # noqa: E402
from d4rt_lingbot.hooks import canvas_affine, to_canvas  # noqa: E402
from lingbotvla.utils.observation_augmentation import ObservationAugmenter  # noqa: E402


def _affine_plan(scale, tx, ty):
    recipe = yaml.safe_load((ROOT / "configs/vla/robotwin/full432_augmentation_recipe.yaml").read_text())
    aug = ObservationAugmenter(recipe)
    plan = aug.sample(0)
    plan.update(keep_original=False, color=False, affine=True, occlusion=False, gaussian_noise=False,
                salt_pepper=False, jpeg=False, blur="none", scale=scale, translation=[tx, ty])
    return aug, plan


def _texture(size):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    f = size / 224.0
    img = np.stack([128 + 90 * np.sin(2 * np.pi * x / (23 * f)) * np.cos(2 * np.pi * y / (31 * f)),
                    128 + 90 * np.cos(2 * np.pi * (x + y) / (37 * f)),
                    128 + 90 * np.sin(2 * np.pi * (x - 0.5 * y) / (29 * f))], axis=-1)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _synthetic_episode(root, task="toy", ep=0, T=60):
    """Fronto-parallel planes at 0.8 / 1.0 / 1.2 m; three cameras with distinct poses."""
    d = pathlib.Path(root) / task
    name = f"episode_{ep:06d}"
    K = np.array([[358.6, 0, 160.0], [0, 358.6, 120.0], [0, 0, 1]])
    extr = np.zeros((3, T, 3, 4))
    for c in range(3):
        a = 0.1 * c
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        extr[c, :, :, :3] = R
        extr[c, :, :, 3] = [0.05 * c, -0.02 * c, 0.1 * c]
    intr = np.broadcast_to(K, (3, T, 3, 3)).copy()
    grip = np.zeros((T, 3, 2, 4))
    grip[:, 0, 0] = [100.0, 130.0, 0.9, 1.0]
    grip[:, 0, 1] = [220.0, 110.0, 0.9, 1.0]
    ee = np.zeros((T, 16), np.float32)
    ee[:, 3] = ee[:, 11] = 1.0  # identity quaternions (w first)
    ee[:, 8:11] = [0.3, 0.0, 0.5]
    for sub in ("ee", "cam") + tuple(f"{k}/{c}" for k in ("depth", "depth_meta", "tracks", "tvis", "tmeta") for c in D.CAMERAS):
        (d / sub).mkdir(parents=True, exist_ok=True)
    np.save(d / "ee" / f"{name}.npy", ee)
    np.savez(d / "cam" / f"{name}.npz", intrinsic=intr, extrinsic=extr, grip=grip)
    gu, gv = np.meshgrid(np.linspace(20, 300, 20), np.linspace(20, 220, 20))
    pts = np.stack([gu.ravel(), gv.ravel()], -1)
    for c, cam in enumerate(D.CAMERAS):
        np.save(d / "depth" / cam / f"{name}.npy", np.full((T, 240, 320), 800 + 200 * c, np.uint16))
        np.savez(d / "depth_meta" / cam / f"{name}.npz", reliable=np.ones(T, bool))
        starts = np.array([0, 10])
        tr = np.stack([np.broadcast_to(pts + np.array([0.3, -0.2]) * np.arange(50)[:, None, None], (50, 400, 2))] * 2)
        np.save(d / "tracks" / cam / f"{name}.npy", tr.astype(np.float16))
        np.save(d / "tvis" / cam / f"{name}.npy", np.ones((2, 50, 400), bool))
        np.savez(d / "tmeta" / cam / f"{name}.npz", starts=starts)
    return task, ep


class CanvasAffineTest(unittest.TestCase):
    def test_matches_observation_augmenter_warp(self):
        src = _texture(384)
        orig224 = to_canvas(src.transpose(2, 0, 1)[None])[0].transpose(1, 2, 0)
        for scale, tx, ty in [(0.9, 0.06, -0.04), (1.1, -0.05, 0.03), (1.0, 0.02, 0.06)]:
            aug, plan = _affine_plan(scale, tx, ty)
            warped, _ = aug.apply({"cam": src}, plan)
            aug224 = to_canvas(warped["cam"].transpose(2, 0, 1)[None])[0].transpose(1, 2, 0)
            a = D.CanvasAffine(scale, tx, ty)
            rc, cc = np.mgrid[30:194, 30:194]
            sr, sc = a.inv_cont(rc + 0.5, cc + 0.5)
            ok = (sr > 8) & (sr < 216) & (sc > 8) & (sc < 216)

            def err(dr, dc):
                ref = cv2.remap(orig224, (sc - 0.5 + dc).astype(np.float32), (sr - 0.5 + dr).astype(np.float32), cv2.INTER_LINEAR)
                return float(np.abs(aug224[rc, cc] - ref)[ok].mean())

            e0 = err(0.0, 0.0)
            self.assertLess(e0, 0.01, (scale, tx, ty))
            for dr, dc in [(0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5)]:  # half-pixel convention errors are visible
                self.assertGreater(err(dr, dc), 2 * e0)

    def test_plan_to_affine(self):
        self.assertIsNone(canvas_affine(None))
        self.assertIsNone(canvas_affine({"keep_original": True}))
        self.assertIsNone(canvas_affine({"keep_original": False, "affine": False, "scale": 0.9, "translation": [0, 0]}))
        a = canvas_affine({"keep_original": False, "affine": True, "scale": 0.95, "translation": [0.01, -0.02]})
        r, c = a.fwd_idx(np.array([0, 111, 223]), np.array([0, 111, 223]))
        np.testing.assert_array_equal(r, np.floor(0.95 * (np.array([0, 111, 223]) + 0.5) + 0.05 * 112 - 0.02 * 224))
        np.testing.assert_array_equal(c, np.floor(0.95 * (np.array([0, 111, 223]) + 0.5) + 0.05 * 112 + 0.01 * 224))

    def test_to_canvas(self):
        x = np.full((3, 3, 384, 384), 200, np.uint8)
        y = to_canvas(x)
        self.assertEqual(y.shape, (3, 3, 224, 224))
        np.testing.assert_allclose(y, 200 / 255, atol=1e-6)
        z = np.random.default_rng(0).random((3, 3, 224, 224)).astype(np.float32)
        np.testing.assert_array_equal(to_canvas(z), z)

    def test_task_names(self):
        self.assertEqual(D.task_from_repo("/a/lerobot/robotwin_clean_place_shoe"), "place_shoe")
        self.assertEqual(D.task_from_repo("/a/b/place_shoe-aloha-agilex_clean_50-50/"), "place_shoe")


class SamplerAffineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.task, self.ep = _synthetic_episode(self.tmp.name)
        self.s = D.D4rtSampler(root=self.tmp.name, num_queries=96)
        self.imgs = np.random.default_rng(1).random((3, 3, 224, 224)).astype(np.float32)

    def tearDown(self):
        self.tmp.cleanup()

    def test_targets_under_augmented_pixels(self):
        offs = np.arange(-7, 7) + 0.5
        n_checked = 0
        for scale, tx, ty in [(0.9, 0.06, -0.04), (1.1, -0.05, 0.03), (1.0, 0.0, 0.0)]:
            a = D.CanvasAffine(scale, tx, ty)
            for frame in range(0, 40, 7):
                out = self.s(self.task, self.ep, frame, self.imgs, a)
                for j in range(self.s.k):
                    if not out["d4rt_mask"][j]:
                        continue
                    r, c, cam = int(out["d4rt_coords"][j, 0]), int(out["d4rt_coords"][j, 1]), int(out["d4rt_cam"][j])
                    self.assertTrue(7 <= r < 217 and 7 <= c < 217)
                    np.testing.assert_array_equal(out["d4rt_patches"][j], self.imgs[cam, :, r - 7:r + 7, c - 7:c + 7] * 2 - 1)
                    if out["d4rt_task"][j] != D.T_DEPTH:
                        continue
                    # depth task: target in c_src @ t -> project -> canvas -> augmented canvas == the patch grid
                    xyz = out["d4rt_target"][j].astype(np.float64) * self.s.xyz_scale
                    K = self.s._store.get(self.task, self.ep)["intr"][cam, frame]
                    u = xyz[..., 0] / xyz[..., 2] * K[0, 0] + K[0, 2]
                    v = xyz[..., 1] / xyz[..., 2] * K[1, 1] + K[1, 2]
                    ar = a.s * (v / D.SY) + a.off_r
                    ac = a.s * (u / D.SX) + a.off_c
                    np.testing.assert_allclose(ar, np.broadcast_to((r + offs)[:, None], ar.shape), atol=1e-4)
                    np.testing.assert_allclose(ac, np.broadcast_to((c + offs)[None, :], ac.shape), atol=1e-4)
                    n_checked += 1
        self.assertGreater(n_checked, 50)

    def test_identity_affine_is_no_affine(self):
        for frame in (0, 13, 30):
            ref = self.s(self.task, self.ep, frame, self.imgs)
            out = self.s(self.task, self.ep, frame, self.imgs, D.CanvasAffine())
            for k in ref:
                np.testing.assert_array_equal(ref[k], out[k], k)


if __name__ == "__main__":
    unittest.main()
