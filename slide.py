#!/usr/bin/env python3
import math
import os
import time
import pybullet as p
import pybullet_data as pd


def _combined_aabb_z(body_id):
    zmin = float("inf")
    zmax = float("-inf")
    for idx in [-1] + list(range(p.getNumJoints(body_id))):
        aabb = p.getAABB(body_id, idx)
        zmin = min(zmin, aabb[0][2])
        zmax = max(zmax, aabb[1][2])
    return zmin, zmax


def auto_upright_orientation(body_id):
    """
    Try several 90-degree rotations and pick the one with the tallest vertical span.
    This keeps the Y-up humanoid URDF standing in the Z-up world.
    """
    candidates = [
        p.getQuaternionFromEuler([0, 0, 0]),
        p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
        p.getQuaternionFromEuler([-math.pi / 2, 0, 0]),
        p.getQuaternionFromEuler([0, math.pi / 2, 0]),
        p.getQuaternionFromEuler([0, -math.pi / 2, 0]),
    ]
    best_q = candidates[0]
    best_height = -1.0
    best_zmin = 0.0
    for q in candidates:
        p.resetBasePositionAndOrientation(body_id, [0, 0, 0], q)
        zmin, zmax = _combined_aabb_z(body_id)
        height = zmax - zmin
        if height > best_height:
            best_height = height
            best_q = q
            best_zmin = zmin
    clearance = 0.01
    base_z = -best_zmin + clearance
    return best_q, base_z


def build_ground(friction=2.0, restitution=0.0):
    """Add a plane and tune contact so feet don't slide or bounce."""
    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(
        plane,
        -1,
        lateralFriction=friction,
        restitution=restitution,
        rollingFriction=0.01,
        spinningFriction=0.01,
    )
    return plane


def main():
    cid = p.connect(p.GUI)
    if cid < 0:
        raise RuntimeError("PyBullet GUI connect failed")
    p.resetSimulation()
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, -9.81)
    build_ground()

    humanoid_rel = os.path.join("humanoid", "humanoid.urdf")
    humanoid_abs = os.path.join(pd.getDataPath(), humanoid_rel)
    if not os.path.exists(humanoid_abs):
        raise FileNotFoundError(f"Missing URDF: {humanoid_abs}")

    humanoid = p.loadURDF(
        humanoid_rel,
        [0, 0, 1.0],
        p.getQuaternionFromEuler([0, 0, 0]),
        flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT,
        useFixedBase=False,
        globalScaling=0.6,
    )

    # Orient the Y-up humanoid to stand upright in the Z-up world and lift it off the ground a bit.
    upright_quat, base_z = auto_upright_orientation(humanoid)
    p.resetBasePositionAndOrientation(humanoid, [0, 0, base_z], upright_quat)

    while p.isConnected():
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
