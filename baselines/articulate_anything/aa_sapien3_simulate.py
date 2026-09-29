#!/usr/bin/env python
"""aa_sapien3_simulate.py -- SAPIEN 3.0.3 port of the official articulate_anything/physics/sapien_simulate.py (plus the
camera / picture helpers it imports from sapien_render.py).

Why: the official pin sapien==2.2.2 segfaults inside svulkan2 (Buffer::map during the first camera render) on the run
node (RTX 5090, driver 595.71); SAPIEN 3.0.3 (already in the campaign base env) renders correctly there. This port keeps
the official hydra config (conf/simulator/default.yaml), the same CLI overrides produced by config_to_command, the same
camera placement/FOV/lighting/floor, the same stationary/move logic, and writes the same files (robot_<view>.png,
robot_<view>_seg[_color].png + seg.json, video_<joint>_<view>.mp4, raise_distances.json). API differences only:
  * Engine/SapienRenderer/load_kinematic -> sapien.Scene + URDFLoader.load (dynamic articulation with fix_root_link and
    gravity disabled, which is what a kinematic articulation amounts to for set_qpos-driven rendering);
  * camera mounting via the camera entity pose instead of a kinematic mount actor + set_parent;
  * camera.get_float_texture("Color") -> camera.get_picture("Color"); get_visual_actor_segmentation() ->
    get_picture("Segmentation")[..., 1]; link.id -> link.entity.per_scene_id;
  * the checkerboard floor (sapien.asset.create_checkerboard, SAPIEN 2 only) is not ported: official default is "plain".
Invoked by the patched make_cmd() in place of sapien_simulate.py (AA_RENDER_SCRIPT). Run from the source tree cwd.
"""
import logging
import os
from functools import partial
from typing import Any, Dict

import cv2
import hydra
import numpy as np
import pybullet as p
import sapien
from omegaconf import DictConfig, OmegaConf
from PIL import ImageColor
from scipy.spatial.transform import Rotation as R

from articulate_anything.physics.pybullet_utils import setup_pybullet
from articulate_anything.physics.sapien_render import (
    VideoWriterManager,
    apply_morphological_operations,
    create_segmentation_image,
    flip_video,
    get_distinct_colors,
    save_image,
)
from articulate_anything.utils.utils import create_dir, join_path, load_json, save_json

AA_SRC = os.environ.get("AA_SRC") or os.getcwd()
CONFIG_PATH = os.path.join(AA_SRC, "conf", "simulator")


# --------------------------------------------------------------------------- sapien_render.py ports
def setup_cameras(cfg, scene):
    cameras = {}
    for view_name, view_params in cfg.camera_params.views.items():
        camera = scene.add_camera(
            name=view_name,
            width=cfg.camera_params.width,
            height=cfg.camera_params.height,
            fovy=np.deg2rad(35),
            near=0.1,
            far=100,
        )
        cam_pos = np.array(view_params.cam_pos)
        look_at = np.array(view_params.look_at)
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera.entity.set_pose(sapien.Pose(mat44))
        cameras[view_name] = camera
    return cameras


def get_segmentation_data(camera):
    camera.take_picture()
    seg_labels = camera.get_picture("Segmentation")  # uint32 (H, W, 4): [mesh id, entity id, 0, 0]
    label_image = seg_labels[..., 1].astype(np.uint8)
    cleaned_label_image = apply_morphological_operations(label_image)
    unique_labels = np.unique(cleaned_label_image)
    return cleaned_label_image, unique_labels


def create_color_mappings(unique_labels, robot):
    distinct_colors = get_distinct_colors(len(unique_labels))
    color_palette = np.array([ImageColor.getrgb(color) for color in distinct_colors], dtype=np.uint8)
    link_id_to_name = {int(link.entity.per_scene_id): link.name for link in robot.get_links()}
    link_id_to_color = {label: color_palette[i % len(distinct_colors)] for i, label in enumerate(unique_labels)}
    link_color_mapping = {
        link_id_to_name[label]: "#{:02x}{:02x}{:02x}".format(*color)
        for label, color in link_id_to_color.items()
        if label in link_id_to_name
    }
    return link_id_to_name, link_id_to_color, link_color_mapping


def take_camera_pic(robot, camera, use_segmentation=True, output_json="seg.json", object_white=True):
    camera.take_picture()
    if not use_segmentation:
        rgba = camera.get_picture("Color")
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        return cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)
    label_image, unique_labels = get_segmentation_data(camera)
    link_id_to_name, link_id_to_color, link_color_mapping = create_color_mappings(unique_labels, robot)
    color_image = create_segmentation_image(label_image, link_id_to_color, link_id_to_name, object_white)
    if object_white:
        link_color_mapping = {k: "#FFFFFF" for k in link_color_mapping}
        link_color_mapping["background"] = "#000000"
    if use_segmentation:
        save_json(link_color_mapping, output_json)
    return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- sapien_simulate.py ports
def make_floor(floor_texture, scene, renderer=None):
    if floor_texture == "plain":
        scene.add_ground(altitude=0)
    elif floor_texture == "checkerboard":
        raise NotImplementedError(
            "checkerboard floor (sapien.asset.create_checkerboard) is SAPIEN-2 only; official default is 'plain'"
        )
    else:
        raise ValueError(f"Texture {floor_texture} not recognized. Supported textures are 'plain' and 'checkerboard'")


def setup_sapien(cfg):
    if cfg.ray_tracing:
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(64)
        sapien.render.set_ray_tracing_denoiser("oidn")

    scene_config = sapien.physx.PhysxSceneConfig()
    scene_config.gravity = [0.0, 0.0, 0.0]  # kinematic-articulation equivalent: joints keep the qpos we set
    sapien.physx.set_scene_config(scene_config)
    scene = sapien.Scene()
    scene.set_timestep(cfg.engine.timestep)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(cfg.urdf.file)
    assert robot, "URDF not loaded."

    if cfg.urdf.raise_distance_file is None:
        cfg.urdf.raise_distance_file = join_path(os.path.dirname(cfg.urdf.file), "raise_distances.json")

    if not os.path.exists(cfg.urdf.raise_distance_file):
        # NOTE (official): the raise distance is computed with pybullet
        client, robot_id = setup_pybullet(cfg.urdf.file)
        cfg.urdf.raise_distance_file = join_path(os.path.dirname(cfg.urdf.file), "raise_distances.json")
        p.disconnect(client)

    raise_distance = load_json(cfg.urdf.raise_distance_file)
    rotation = R.from_euler("xyz", [cfg.urdf.rotation_pose.rx, cfg.urdf.rotation_pose.ry, cfg.urdf.rotation_pose.rz])
    quat = rotation.as_quat()  # [x, y, z, w]
    quat = [quat[3], quat[0], quat[1], quat[2]]  # Sapien convention: wxyz
    pose = sapien.Pose(p=[0, 0, max(raise_distance) + cfg.urdf.raise_distance_offset], q=quat)
    robot.set_pose(pose)

    make_floor(cfg.floor_texture, scene)
    scene.set_ambient_light(cfg.lighting.ambient)
    scene.add_directional_light(cfg.lighting.directional.direction, cfg.lighting.directional.color)
    return None, scene, robot


def get_manipulatable_joints(robot):
    """Same output as the official function: [(name, type, (lower, upper))] in qpos order."""
    manipulatable_joints = []
    for joint in robot.get_active_joints():
        jtype = joint.type
        if jtype in ("revolute_unwrapped", "continuous"):
            jtype = "revolute"
        if jtype not in ("revolute", "prismatic"):
            raise ValueError("active joint %s of unsupported type %s" % (joint.name, joint.type))
        limits = np.asarray(joint.get_limits())[0]
        if jtype == "revolute" and limits[0] == -float("inf") and limits[1] == float("inf"):
            new_limits = (0, 4.89)  # continuous joint: approximately 0 to 280 degrees (official)
        else:
            new_limits = (float(limits[0]), float(limits[1]))
        manipulatable_joints.append((joint.name, jtype, new_limits))
    return manipulatable_joints


def get_joint_idx_by_name(manipulatable_joints, joint_name):
    for i, joint in enumerate(manipulatable_joints):
        if joint[0] == joint_name:
            return i
    return None


def move_joint(cfg: DictConfig, scene, cameras, robot, joint_name, joints_dict, video_writers):
    joint_idx = get_joint_idx_by_name(joints_dict, joint_name)
    if joint_idx is None:
        raise ValueError(f"Joint name '{joint_name}' not found")
    _, _, (lower_limit, upper_limit) = joints_dict[joint_idx]

    def move_up(num_steps, step):
        return lower_limit + step * (upper_limit - lower_limit) / num_steps

    def move_down(num_steps, step):
        return upper_limit - step * (upper_limit - lower_limit) / num_steps

    move_func_dict = {"auto": move_down if lower_limit < 0 else move_up, "move_up": move_up, "move_down": move_down}
    move_func = move_func_dict[cfg.simulation_params.joint_move_dir]
    compute_target_pos = partial(move_func, cfg.simulation_params.num_steps)

    trajectory = []
    for step in range(cfg.simulation_params.num_steps):
        qpos = robot.get_qpos()
        qpos[joint_idx] = compute_target_pos(step)
        robot.set_qpos(qpos)
        scene.step()
        scene.update_render()
        for camera_name, camera in cameras.items():
            bgr_img = take_camera_pic(
                robot,
                camera,
                use_segmentation=cfg.use_segmentation,
                output_json=cfg.output.seg_json,
                object_white=cfg.object_white,
            )
            video_writers[camera_name].write(bgr_img)
        trajectory.append(np.asarray(qpos).tolist())

    if cfg.flip_video:
        for camera_name, video_writer in video_writers.items():
            video_writer.release()
            video_path = os.path.join(cfg.output.dir, f"video_{joint_name}_{camera_name}.mp4")
            flip_video(video_path)
    return trajectory


def simulate_sapien(cfg: DictConfig):
    logging.info(
        f">>>> Generating a {cfg.simulation_params.stationary_or_move} of joint: {cfg.simulation_params.joint_name} | file: {cfg.urdf.file}"
    )
    if cfg.output.dir is None:
        cfg.output.dir = join_path(os.path.dirname(cfg.urdf.file))
    if cfg.output.seg_json is None:
        cfg.output.seg_json = join_path(cfg.output.dir, "seg.json")
    create_dir(cfg.output.dir)
    _, scene, robot = setup_sapien(cfg)
    manipulatable_joints = get_manipulatable_joints(robot)
    cameras = setup_cameras(cfg, scene)

    if cfg.simulation_params.stationary_or_move == "move":
        capture_video(cfg, scene, robot, cameras, manipulatable_joints)
    elif cfg.simulation_params.stationary_or_move == "stationary":
        capture_photo(cfg, scene, robot, cameras)
    else:
        raise ValueError(
            f"Stationary or move option '{cfg.simulation_params.stationary_or_move}' not recognized. Must be 'move' or 'stationary'."
        )
    if not cfg.headless:
        raise NotImplementedError("viewer mode is not available in the headless run")


def capture_video(cfg: DictConfig, scene, robot, cameras: dict, manipulatable_joints):
    if cfg.simulation_params.joint_name != "all" and cfg.simulation_params.joint_name not in [
        joint[0] for joint in manipulatable_joints
    ]:
        raise ValueError(
            f"Joint name '{cfg.simulation_params.joint_name}' not found in the list of manipulatable joints of the URDF file. {manipulatable_joints}"
        )
    initial_qpos = robot.get_qpos()
    for joint_name, joint_type, (lower_limit, upper_limit) in manipulatable_joints:
        robot.set_qpos(initial_qpos)
        scene.step()
        scene.update_render()
        if cfg.simulation_params.joint_name != "all" and cfg.simulation_params.joint_name != joint_name:
            continue
        with VideoWriterManager(cfg, joint_name) as video_writers:
            move_joint(cfg, scene, cameras, robot, joint_name, manipulatable_joints, video_writers)


def get_photo_name(cfg, camera_name):
    if not cfg.use_segmentation:
        return f"robot_{camera_name}.png"
    if cfg.object_white:
        return f"robot_{camera_name}_seg.png"
    return f"robot_{camera_name}_seg_color.png"


def capture_photo(cfg: DictConfig, scene, robot, cameras: Dict[str, Any]):
    set_joint_to_target_limit(robot, cfg.simulation_params.joint_move_dir)
    scene.step()
    scene.update_render()
    for camera_name, camera in cameras.items():
        bgr_img = take_camera_pic(
            robot,
            camera,
            use_segmentation=cfg.use_segmentation,
            output_json=cfg.output.seg_json,
            object_white=cfg.object_white,
        )
        photo_path = join_path(cfg.output.dir, get_photo_name(cfg, camera_name))
        save_image(bgr_img, photo_path)


def set_joint_to_target_limit(robot, joint_move_dir):
    manipulatable_joints = get_manipulatable_joints(robot)
    qpos = robot.get_qpos()
    for joint_idx, (joint_name, joint_type, (lower_limit, upper_limit)) in enumerate(manipulatable_joints):
        if joint_move_dir == "auto":
            target_limit = upper_limit if lower_limit < 0 else lower_limit
        elif joint_move_dir == "move_up":
            target_limit = lower_limit
        elif joint_move_dir == "move_down":
            target_limit = upper_limit
        else:
            raise ValueError(f"Invalid joint_move_dir: {joint_move_dir}. Must be 'auto', 'move_up', or 'move_down'.")
        qpos[joint_idx] = target_limit
    robot.set_qpos(qpos)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="default")
def main(cfg: DictConfig):
    simulate_sapien(cfg)


if __name__ == "__main__":
    main()
