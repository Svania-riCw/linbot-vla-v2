"""Export and CPU-check a dual-arm RoboTwin FK profile in the simulator env.

Uses the existing embodiment config/URDF. Does not launch a renderer, policy,
training job, or alter a live environment.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.utils.ee_roi import DualArmFK, canonical_hash
from lingbotvla.utils.roi_profile import export_robotwin_profile


def export(robot_dir):
    import sapien.core as sapien

    robot_dir = Path(robot_dir).resolve()
    cfg = yaml.safe_load((robot_dir / "config.yml").read_text(encoding="utf-8"))
    poses = cfg["robot_pose"]
    if len(poses) > 1 and not np.allclose(poses[0], poses[1]):
        raise ValueError("This exporter requires one fixed-root dual-arm articulation")
    if hasattr(sapien, "physx"):
        # SAPIEN 3's default Scene also creates a Vulkan renderer.
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    else:
        engine = sapien.Engine()
        scene = engine.create_scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # Keep all joint origins, axes, limits and inertials from the real URDF.
    # Mesh visuals/collisions are irrelevant to FK and otherwise initialize a
    # renderer even when only the CPU physics system is requested.
    tree = ET.parse(robot_dir / cfg["urdf_path"])
    for link in tree.getroot().findall("link"):
        for child in list(link):
            if child.tag in ("visual", "collision"):
                link.remove(child)
    with tempfile.TemporaryDirectory(prefix="roi_fk_") as temporary:
        kinematic_urdf = Path(temporary) / "robot.urdf"
        tree.write(kinematic_urdf, encoding="utf-8", xml_declaration=True)
        entity = loader.load(str(kinematic_urdf))
    if entity is None:
        raise ValueError("SAPIEN could not load the embodiment URDF")
    base = sapien.Pose(poses[0][:3], poses[0][3:])
    entity.set_root_pose(base)
    robot = SimpleNamespace()
    for side, index in (("left", 0), ("right", 1)):
        for key, value in dict(entity=entity, entity_origion_pose=base,
                arm_joints_name=cfg["arm_joints_name"][index],
                ee=entity.find_joint_by_name(cfg["ee_joints"][index]),
                gripper_bias=cfg["gripper_bias"],
                global_trans_matrix=cfg.get("global_trans_matrix", np.eye(3)),
                delta_matrix=cfg.get("delta_matrix", np.eye(3))).items():
            setattr(robot, f"{side}_{key}", value)
    profile = export_robotwin_profile(robot)
    fk = DualArmFK(profile)
    names = [joint.get_name() for joint in entity.get_active_joints()]
    rng = np.random.default_rng(42)
    errors = []
    for trial in range(8):
        state = np.zeros(14) if trial == 0 else rng.uniform(-0.4, 0.4, size=14)
        qpos = np.zeros(len(names))
        for side, start in (("left", 0), ("right", 7)):
            for index, name in enumerate(getattr(robot, f"{side}_arm_joints_name")):
                qpos[names.index(name)] = state[start+index]
        entity.set_qpos(qpos)
        expected = []
        for side in ("left", "right"):
            pose = getattr(robot, f"{side}_ee").get_global_pose().to_transformation_matrix()
            expected.append(pose[:3, :3] @ profile["arms"][side]["tip_offset"] + pose[:3, 3])
        errors.append(float(np.max(np.linalg.norm(fk.positions(state) - expected, axis=-1))))
    if max(errors) > 1e-5:
        raise ValueError(f"FK differs from SAPIEN joint frames: {max(errors)} metres")
    return profile, dict(max_error_m=max(errors), samples=len(errors),
                         profile_sha256=canonical_hash(profile))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new profile path")
    profile, report = export(args.robot_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    args.output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
