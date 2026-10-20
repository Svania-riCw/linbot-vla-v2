"""Export the kinematic subset of a RoboTwin URDF-loaded SAPIEN robot."""
import numpy as np

from .ee_roi import DualArmFK, ROIGeometryError


def export_robotwin_profile(robot):
    profile = dict(schema_version=1, state_dim=14, state_semantics="robotwin_drive_target", arms={})
    for side, start in (("left", 0), ("right", 7)):
        entity = getattr(robot, f"{side}_entity")
        names = list(getattr(robot, f"{side}_arm_joints_name"))
        if len(names) != 6:
            raise ROIGeometryError("ROI-432 currently supports the 6+gripper+6+gripper robot only")
        indices = {name: start+i for i, name in enumerate(names)}
        tip = getattr(robot, f"{side}_ee")
        by_child = {j.get_child_link().get_name(): j for j in entity.get_joints()
                    if j.get_parent_link() is not None}
        # SAPIEN joint.global_pose is the moving child-side joint frame. Include
        # the tip joint itself, then return from its child link to that frame.
        chain, link, visited = [], tip.get_child_link(), set()
        while link.get_name() in by_child:
            if link.get_name() in visited:
                raise ROIGeometryError("Cycle in articulation")
            visited.add(link.get_name())
            joint = by_child[link.get_name()]
            name, kind = joint.get_name(), str(joint.type)
            if kind == "revolute_unwrapped":
                kind = "revolute"
            if kind != "fixed" and name not in indices:
                raise ROIGeometryError(f"Unobserved movable joint in {side} chain: {name}")
            chain.append(dict(name=name, type=kind, state_index=indices.get(name),
                              parent_from_joint=joint.get_pose_in_parent().to_transformation_matrix().tolist(),
                              child_from_joint=joint.get_pose_in_child().to_transformation_matrix().tolist()))
            link = joint.get_parent_link()
        # Match Robot._trans_endpose(..., is_endpose=False), which populates
        # get_obs()['endpose']; the offset is in the actual ee joint frame.
        bias = float(getattr(robot, f"{side}_gripper_bias")) - 0.12
        offset = (np.asarray(getattr(robot, f"{side}_global_trans_matrix"))
                  @ np.asarray(getattr(robot, f"{side}_delta_matrix"))
                  @ np.array([bias, 0, 0]))
        base = getattr(robot, f"{side}_entity_origion_pose")
        profile["arms"][side] = dict(chain=chain[::-1],
            parent_from_tip=tip.get_pose_in_child().to_transformation_matrix().tolist(),
            tip_offset=offset.tolist(), world_from_base=base.to_transformation_matrix().tolist())
    DualArmFK(profile)  # check all six observed joints, transforms and dimensions
    return profile
