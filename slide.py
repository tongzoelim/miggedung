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
    # 발이 바닥에 닿도록 음의 clearance 설정
    clearance = -0.02  # 약간의 음수값으로 설정하여 바닥에 닿게 함
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


def axis_label(axis_vec, tol=0.3):
    """Map joint axis vector to dominant axis label."""
    ax = [abs(axis_vec[0]), abs(axis_vec[1]), abs(axis_vec[2])]
    idx = ax.index(max(ax))
    if ax[idx] < tol:
        return None
    return "xyz"[idx]


def map_humanoid_joints(robot_id):
    """
    Deterministic map for the built-in PyBullet humanoid.
    Right hip/knee/ankle: 3/4/5, Left hip/knee/ankle: 9/10/11.
    Yaw/roll stay None (legs are single-DOF pitch hinges).
    """
    return {
        "left": {"hip": {"yaw": None, "roll": None, "pitch": 9}, "knee": 10, "ankle_pitch": 11, "ankle_roll": None},
        "right": {"hip": {"yaw": None, "roll": None, "pitch": 3}, "knee": 4, "ankle_pitch": 5, "ankle_roll": None},
    }


def get_foot_links(robot_id):
    """Find link indices that look like feet/ankles by name heuristics."""
    candidates = {"left": None, "right": None}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        link_name = info[12].decode("utf-8").lower()
        if any(k in link_name for k in ["foot", "ankle", "sole"]):
            if "left" in link_name or "l_" in link_name:
                candidates["left"] = j
            if "right" in link_name or "r_" in link_name:
                candidates["right"] = j
    return candidates


def disable_default_motors(robot_id):
    """Set all controllable joints to zero-velocity mode to avoid built-in motors."""
    indices = []
    zeros = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        jt = info[2]
        if jt in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            indices.append(j)
            zeros.append(0.0)
    if indices:
        p.setJointMotorControlArray(
            robot_id,
            indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=zeros,
            forces=zeros,
        )


def change_foot_friction(robot_id, foot_links, mu=2.5):
    """Raise lateral friction on feet."""
    for link in foot_links.values():
        if link is not None:
            p.changeDynamics(
                robot_id,
                link,
                lateralFriction=mu,
                rollingFriction=0.01,
                spinningFriction=0.01,
                restitution=0.0,
            )


def damp_all_links(robot_id, lin=0.05, ang=0.05):
    """Light damping to reduce jitter."""
    p.changeDynamics(robot_id, -1, linearDamping=lin, angularDamping=ang)
    for j in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, j, linearDamping=lin, angularDamping=ang)


def add_debug_sliders():
    sliders = {
        "speed_hz": p.addUserDebugParameter("speed_hz", 0.2, 1.8, 0.55),
        "step_len": p.addUserDebugParameter("step_len", 0.1, 1.0, 0.50),
        "step_height": p.addUserDebugParameter("step_height", 0.1, 0.8, 0.50),
        "hip_sway": p.addUserDebugParameter("hip_sway", 0.0, 0.7, 0.16),
        "lean_fwd": p.addUserDebugParameter("lean_fwd", -0.1, 0.35, 0.12),
        "foot_friction": p.addUserDebugParameter("foot_friction", 0.5, 4.5, 3.0),
        "stand_kp": p.addUserDebugParameter("stand_kp", 0.5, 4.0, 2.0),
        "stand_kd": p.addUserDebugParameter("stand_kd", 0.01, 0.4, 0.08),
    }
    return sliders


def read_sliders(sliders):
    return {
        "speed_hz": p.readUserDebugParameter(sliders["speed_hz"]),
        "step_len": p.readUserDebugParameter(sliders["step_len"]),
        "step_height": p.readUserDebugParameter(sliders["step_height"]),
        "hip_sway": p.readUserDebugParameter(sliders["hip_sway"]),
        "lean_fwd": p.readUserDebugParameter(sliders["lean_fwd"]),
        "foot_friction": p.readUserDebugParameter(sliders["foot_friction"]),
        "stand_kp": p.readUserDebugParameter(sliders["stand_kp"]),
        "stand_kd": p.readUserDebugParameter(sliders["stand_kd"]),
    }


def set_initial_standing_pose(robot_id, joint_map, base_pos=(0, 0, 1.0), base_quat=(0, 0, 0, 1)):
    """Reset base and joints into a balanced stand pose."""
    p.resetBasePositionAndOrientation(robot_id, base_pos, base_quat)
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
    # 완전히 똑바로 선 자세 (모든 관절 0도)
    stand = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": 0.0, "knee": 0.0, "ankle_pitch": 0.0, "ankle_roll": 0.0}
    for side in ("left", "right"):
        leg = joint_map[side]
        sign = +1.0 if side == "left" else -1.0
        hip_yaw_id = leg["hip"].get("yaw")
        hip_roll_id = leg["hip"].get("roll")
        hip_pitch_id = leg["hip"].get("pitch")
        if hip_pitch_id is not None:
            jt = p.getJointInfo(robot_id, hip_pitch_id)[2]
            if jt == p.JOINT_SPHERICAL:
                roll = sign * stand["hip_roll"]
                pitch = stand["hip_pitch"]
                yaw = stand["hip_yaw"]
                quat = p.getQuaternionFromEuler([roll, pitch, yaw])
                p.resetJointStateMultiDof(robot_id, hip_pitch_id, targetValue=quat)
            else:
                if hip_yaw_id is not None:
                    p.resetJointState(robot_id, hip_yaw_id, stand["hip_yaw"], 0.0)
                if hip_roll_id is not None:
                    p.resetJointState(robot_id, hip_roll_id, sign * stand["hip_roll"], 0.0)
                p.resetJointState(robot_id, hip_pitch_id, stand["hip_pitch"], 0.0)
        if leg["knee"] is not None:
            p.resetJointState(robot_id, leg["knee"], stand["knee"], 0.0)
        ankle_pitch_id = leg["ankle_pitch"]
        ankle_roll_id = leg["ankle_roll"]
        if ankle_pitch_id is not None:
            jt = p.getJointInfo(robot_id, ankle_pitch_id)[2]
            if jt == p.JOINT_SPHERICAL:
                roll = sign * stand["ankle_roll"]
                pitch = stand["ankle_pitch"]
                quat = p.getQuaternionFromEuler([roll, pitch, 0.0])
                p.resetJointStateMultiDof(robot_id, ankle_pitch_id, targetValue=quat)
            else:
                p.resetJointState(robot_id, ankle_pitch_id, stand["ankle_pitch"], 0.0)
                if ankle_roll_id is not None:
                    p.resetJointState(robot_id, ankle_roll_id, sign * stand["ankle_roll"], 0.0)
        elif ankle_roll_id is not None:
            p.resetJointState(robot_id, ankle_roll_id, sign * stand["ankle_roll"], 0.0)


def set_initial_arm_pose(robot_id, shoulder_pitch=-1.2, elbow_bend=0.3):
    """Lower arms from T-pose to the side with a slight elbow bend."""
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[1].decode("utf-8").lower()
        jt = info[2]
        axis = info[13]
        if "shoulder" in name:
            # Aim arms downward along the body; mainly pitch axis.
            lbl = axis_label(axis)
            if jt == p.JOINT_SPHERICAL:
                quat = p.getQuaternionFromEuler([0.0, shoulder_pitch, 0.0])
                try:
                    p.resetJointStateMultiDof(robot_id, j, targetValue=quat)
                except TypeError:
                    p.resetJointStateMultiDof(robot_id, j, quat)
            elif jt == p.JOINT_REVOLUTE and lbl == "y":
                p.resetJointState(robot_id, j, shoulder_pitch, 0.0)
            elif jt == p.JOINT_REVOLUTE:
                # If axis unknown, still try lowering with the pitch value.
                p.resetJointState(robot_id, j, shoulder_pitch, 0.0)
        if "elbow" in name and jt == p.JOINT_REVOLUTE:
            p.resetJointState(robot_id, j, elbow_bend, 0.0)


def hold_stand_pose(robot_id, joint_map, force=900):
    """로봇이 완전히 정지하도록 제어"""
    for side in ("left", "right"):
        leg = joint_map[side]
        
        # 모든 관절에 대해 0 속도 제어 적용
        for joint_type in ["yaw", "roll", "pitch"]:
            joint_id = leg["hip"].get(joint_type)
            if joint_id is not None:
                p.setJointMotorControl2(
                    robot_id, 
                    joint_id, 
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=force * 0.3
                )
        
        # 무릎 관절
        if leg["knee"] is not None:
            p.setJointMotorControl2(
                robot_id, 
                leg["knee"], 
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=force * 0.3
            )
            
        # 발목 관절
        for joint_type in ["ankle_pitch", "ankle_roll"]:
            joint_id = leg.get(joint_type)
            if joint_id is not None:
                p.setJointMotorControl2(
                    robot_id,
                    joint_id,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=force * 0.3
                )


class HumanoidWalker:
    def __init__(self, robot_id, joint_map):
        self.id = robot_id
        self.jm = joint_map
        self.joint_types = {j: p.getJointInfo(robot_id, j)[2] for j in range(p.getNumJoints(robot_id))}
        # Neutral pose for gait generation (slightly bent knee to avoid stiff-leg impacts)
        # Knee flexion is negative (limits: [-3.14, 0]). Start slightly bent.
        self.neutral = {
            "hip_yaw": 0.0,
            "hip_roll": 0.0,
            "hip_pitch": -0.12,
            "knee": -0.25,
            "ankle_pitch": -0.16,
            "ankle_roll": 0.0,
        }
        self.tau_max = {
            "hip_yaw": 120.0,
            "hip_roll": 220.0,
            "hip_pitch": 260.0,
            "knee": 260.0,
            "ankle_pitch": 160.0,
            "ankle_roll": 140.0,
        }

    def _des_angles_leg(self, t, side, gait, amp, base_rpy):
        f = gait["speed_hz"]
        step_len = gait["step_len"]
        step_h = gait["step_height"]
        sway = gait["hip_sway"]
        lean = gait["lean_fwd"]

        phi = 2.0 * math.pi * f * t + (0.0 if side == "left" else math.pi)
        s_raw = math.sin(phi)
        s = amp * s_raw
        c = math.cos(phi)
        pos = 0.5 * (s + abs(s))   # swing (forward)
        neg = 0.5 * (-s + abs(s))  # stance (backward)

        # Forward/back sway for a smoother human-like heel-strike → toe-off
        hip_pitch = self.neutral["hip_pitch"] - lean + (step_len * 0.90) * s + 0.05 * c
        hip_roll = self.neutral["hip_roll"] + (sway * 0.55 * amp) * math.sin(phi + (math.pi / 2.0))
        # Stance knee stays a bit flexed; swing knee lifts more
        # Lift swing leg higher for toe clearance (bigger knee tuck, stance knee flexed).
        # Knee flexion is negative, so subtract to bend.
        knee = self.neutral["knee"] - 0.30 * neg - (1.90 * step_h + 0.50) * pos
        # Push-off on stance, dorsiflex during swing for toe clearance
        push_off = 0.18 * neg * max(0.0, -s_raw)
        swing_dorsi = 0.28 * (0.5 * (s_raw + abs(s_raw)))
        ankle_pitch = (
            self.neutral["ankle_pitch"]
            - 0.65 * (knee - self.neutral["knee"])
            - 0.14 * s
            - push_off
            + swing_dorsi
            + 0.05 * pos  # extra lift during swing
        )
        # Small yaw to add leg swing arc
        hip_yaw = self.neutral["hip_yaw"] + 0.12 * s
        ankle_roll = self.neutral["ankle_roll"] - 0.22 * hip_roll

        roll, pitch, _ = base_rpy
        hip_roll += -0.6 * roll
        hip_pitch += -0.6 * pitch
        ankle_pitch += -0.9 * pitch
        ankle_roll += -0.6 * roll

        return {
            "hip_yaw": hip_yaw,
            "hip_roll": hip_roll if side == "left" else -hip_roll,
            "hip_pitch": hip_pitch,
            "knee": knee,
            "ankle_pitch": ankle_pitch,
            "ankle_roll": ankle_roll if side == "left" else -ankle_roll,
        }

    def step(self, t, gait, amp=1.0):
        _, base_quat = p.getBasePositionAndOrientation(self.id)
        base_rpy = p.getEulerFromQuaternion(base_quat)
        spherical_targets = {}
        for side in ("left", "right"):
            des = self._des_angles_leg(t, side, gait, amp, base_rpy)
            leg = self.jm[side]
            for role_key, jname in (("hip_yaw", "yaw"), ("hip_roll", "roll"), ("hip_pitch", "pitch")):
                j = leg["hip"].get(jname)
                if j is None:
                    continue
                if self.joint_types.get(j) == p.JOINT_SPHERICAL:
                    tgt = spherical_targets.setdefault(j, {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
                    if role_key == "hip_roll":
                        tgt["roll"] = des[role_key]
                    elif role_key == "hip_pitch":
                        tgt["pitch"] = des[role_key]
                    else:
                        tgt["yaw"] = des[role_key]
                else:
                    p.setJointMotorControl2(
                        self.id, j, p.POSITION_CONTROL,
                        targetPosition=des[role_key],
                        positionGain=2.0,
                        velocityGain=0.15,
                        force=self.tau_max[role_key],
                    )
            if leg["knee"] is not None:
                p.setJointMotorControl2(
                    self.id, leg["knee"], p.POSITION_CONTROL,
                    targetPosition=des["knee"],
                    positionGain=2.2,
                    velocityGain=0.15,
                    force=self.tau_max["knee"],
                )
            if leg["ankle_pitch"] is not None:
                j = leg["ankle_pitch"]
                if self.joint_types.get(j) == p.JOINT_SPHERICAL:
                    tgt = spherical_targets.setdefault(j, {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
                    tgt["pitch"] = des["ankle_pitch"]
                else:
                    p.setJointMotorControl2(
                        self.id, j, p.POSITION_CONTROL,
                        targetPosition=des["ankle_pitch"],
                        positionGain=2.4,
                        velocityGain=0.14,
                        force=self.tau_max["ankle_pitch"],
                    )
            if leg["ankle_roll"] is not None:
                j = leg["ankle_roll"]
                if self.joint_types.get(j) == p.JOINT_SPHERICAL:
                    tgt = spherical_targets.setdefault(j, {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
                    tgt["roll"] = des["ankle_roll"]
                else:
                    p.setJointMotorControl2(
                        self.id, j, p.POSITION_CONTROL,
                        targetPosition=des["ankle_roll"],
                        positionGain=2.2,
                        velocityGain=0.14,
                        force=self.tau_max["ankle_roll"],
                    )
        for j, ang in spherical_targets.items():
            roll = ang.get("roll", 0.0)
            pitch = ang.get("pitch", 0.0)
            yaw = ang.get("yaw", 0.0)
            quat = p.getQuaternionFromEuler([roll, pitch, yaw])
            is_hip = False
            for side in ("left", "right"):
                if j in self.jm[side]["hip"].values():
                    is_hip = True
            force_val = 240 if is_hip else 160
            gain = 1.4 if is_hip else 1.1
            try:
                p.setJointMotorControlMultiDof(
                    self.id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=quat,
                    targetVelocity=[0, 0, 0],
                    positionGain=gain,
                    velocityGain=0.1,
                    force=[force_val, force_val, force_val],
                )
            except TypeError:
                p.setJointMotorControlMultiDof(
                    self.id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=quat,
                    targetVelocity=[0, 0, 0],
                    force=[force_val, force_val, force_val],
                )


def main():
    cid = p.connect(p.GUI)
    if cid < 0:
        raise RuntimeError("PyBullet GUI connect failed")
    p.resetSimulation()
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    plane_id = build_ground()

    humanoid_rel = os.path.join("humanoid", "humanoid.urdf")
    humanoid_abs = os.path.join(pd.getDataPath(), humanoid_rel)
    if not os.path.exists(humanoid_abs):
        raise FileNotFoundError(f"Missing URDF: {humanoid_abs}")

    humanoid = p.loadURDF(
        humanoid_rel,   
        [0, 0, 0.7],  # 초기 높이를 더 낮춤
        p.getQuaternionFromEuler([0, 0, 0]),
        # Keep original joint order so knee indices match known defaults.
        flags=p.URDF_MAINTAIN_LINK_ORDER | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT,
        useFixedBase=False,
        globalScaling=0.6,
    )

    # Orient the Y-up humanoid to stand upright in the Z-up world and lift it off the ground a bit.
    upright_quat, base_z = auto_upright_orientation(humanoid)
    p.resetBasePositionAndOrientation(humanoid, [0, 0, base_z], upright_quat)
    joint_map = map_humanoid_joints(humanoid)
    foot_links = get_foot_links(humanoid)
    change_foot_friction(humanoid, foot_links, mu=4.5)
    damp_all_links(humanoid, lin=0.04, ang=0.04)
    set_initial_standing_pose(humanoid, joint_map, base_pos=[0, 0, base_z], base_quat=upright_quat)
    set_initial_arm_pose(humanoid)
    disable_default_motors(humanoid)
    walker = HumanoidWalker(humanoid, joint_map)  # walking enabled
    sliders = add_debug_sliders()
    last_friction = 3.0
    warmup_time = 3.0
    start_t = time.time()

    while p.isConnected():
        now = time.time()
        t = now - start_t
        gait = read_sliders(sliders)

        # Update foot/ground friction if slider changed.
        if abs(gait["foot_friction"] - last_friction) > 1e-3:
            change_foot_friction(humanoid, foot_links, mu=gait["foot_friction"])
            p.changeDynamics(plane_id, -1, lateralFriction=gait["foot_friction"])
            last_friction = gait["foot_friction"]

        # During warmup, fix base to help stabilize
        if t < warmup_time:
            base_pos, base_quat = p.getBasePositionAndOrientation(humanoid)
            p.resetBasePositionAndOrientation(humanoid, base_pos, base_quat)
            p.resetBaseVelocity(humanoid, [0, 0, 0], [0, 0, 0])

        # Apply walking control if past warmup time
        if t >= warmup_time:
            # Apply walking control
            walker.step(t, gait, amp=1.0)
        else:
            # Keep a standing pose during warmup
            hold_stand_pose(humanoid, joint_map, force=1500)
            
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()