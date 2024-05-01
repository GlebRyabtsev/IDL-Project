import math
import random
import time

import numpy as np
import pkg_resources
import pybullet as p
import pybullet_data

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from scipy.spatial.transform import Rotation as R


class Aviary(BaseRLAviary):
    Kp = np.array((-0.2, -0.2, -0.04))
    Kd = np.array((-0.06, -0.06, -0.01))

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 seed: int = 0,
                 episode_length: int = 8,
                 initial_pos = None
                 ):

        # self.Kp = np.array((-4.0e-3, -4.0e-3, -8.0e-5))
        # self.Kd = np.array((-1.2e-4, -1.2e-4, -2.0e-5))
        # self.INITIAL_POS_RANGE = ((-1.0, 1.0), (-1.0, 1.0), (1.0, 3.0))
        self.INITIAL_ANG_RANGE = ((-0.0, 0.0), (-0.0, 0.0), (0 ,0))

        self.INITIAL_VEL_RANGE = ((-.0, .0),
                                  (-.0, .0),
                                  (-.0, .0))
        # self.INITIAL_ANG_VEL_RANGE = ((-1.0, 1.0),
        #                               (-1.0, 1.0),
        #                               (-1.0, 1.0))
        self.INITIAL_ANG_VEL_RANGE = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        self.TARGET_POS = np.array((0.0, 0.0, 2.0))
        self.EPISODE_LEN_SEC = episode_length
        self._rng = np.random.default_rng(seed)
        self.initial_pos = initial_pos

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=Physics.PYB,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    # def _dynamics(self,
    #               rpm,
    #               nth_drone
    #               ):
    #     """Explicit dynamics implementation.
    #
    #     Based on code written at the Dynamic Systems Lab by James Xu.
    #
    #     Parameters
    #     ----------
    #     rpm : ndarray
    #         (4)-shaped array of ints containing the RPMs values of the 4 motors.
    #     nth_drone : int
    #         The ordinal number/position of the desired drone in list self.DRONE_IDS.
    #
    #     """
    #     #### Current state #########################################
    #     pos = self.pos[nth_drone, :]
    #     quat = self.quat[nth_drone, :]
    #     vel = self.vel[nth_drone, :]
    #     rpy_rates = self.rpy_rates[nth_drone, :]
    #     rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    #     #### Compute forces and torques ############################
    #     forces = np.array(rpm ** 2) * self.KF
    #     thrust = np.array([0, 0, np.sum(forces)])
    #     thrust_world_frame = np.dot(rotation, thrust)
    #     force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
    #     z_torques = np.array(rpm ** 2) * self.KM
    #     if self.DRONE_MODEL == DroneModel.RACE:
    #         z_torques = -z_torques
    #     z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
    #     if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
    #         x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
    #         y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
    #     elif self.DRONE_MODEL == DroneModel.CF2P:
    #         x_torque = (forces[1] - forces[3]) * self.L
    #         y_torque = (-forces[0] + forces[2]) * self.L
    #     torques = np.array([x_torque, y_torque, z_torque])
    #
    #     torques = np.array([0.0, 0.0, 0.0])
    #
    #     torques_corr, thrust_corr = self._get_corrections()
    #     torques += torques_corr
    #     # thrust += thrust_corr
    #     # hover compensation is corrected in preprocessAction
    #     torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
    #     rpy_rates_deriv = np.dot(self.J_INV, torques)
    #     no_pybullet_dyn_accs = force_world_frame / self.M
    #     #### Update state ##########################################
    #     vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
    #     rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
    #     pos = pos + self.PYB_TIMESTEP * vel
    #     quat = self._integrateQ(quat, rpy_rates, self.PYB_TIMESTEP)
    #     #### Set PyBullet's state ##################################
    #     p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
    #                                       pos,
    #                                       quat,
    #                                       physicsClientId=self.CLIENT
    #                                       )
    #     #### Note: the base's velocity only stored and not used ####
    #     p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
    #                         vel,
    #                         np.dot(rotation, rpy_rates),
    #                         physicsClientId=self.CLIENT
    #                         )
    #     #### Store the roll, pitch, yaw rates for the next step ####
    #     self.rpy_rates[nth_drone, :] = rpy_rates
    #
    # def _get_corrections(self):
    #     state = self._getDroneStateVector(0)
    #     quat = state[3:7]
    #     rpy = state[7:10]
    #     ang_vel = state[13:16]
    #     rot = R.from_euler('zyx', rpy)
    #     rotvec = rot.as_rotvec()
    #     torque = self.Kp * rot.apply(rotvec, inverse=True) + self.Kd * rot.apply(ang_vel, inverse=True)
    #
    #     z_unit = np.array([0, 0, 1])
    #
    #     thrust = self.M * 9.81 / (abs(rot.apply(z_unit).dot(z_unit)))
    #
    #     return torque, thrust

    def _preprocessAction(self, action):
        self.action_buffer.append(action)
        rpm = np.zeros((1, 4))
        rpm[0] = np.array(self.HOVER_RPM * (1 + 0.05 * action[0]))  # todo: this coeff is sus
        return rpm

    def _sample_initial_state(self):
        rpy = [self._rng.random() * (upper - lower) + lower for (lower, upper) in self.INITIAL_ANG_RANGE]
        theta = self._rng.random() * (2 * math.pi)
        phi = self._rng.random() * (2 * math.pi)
        dx = math.sin(theta)
        dy = math.cos(theta)
        dz = math.sin(phi)
        if not self.initial_pos:
            pos = [self.TARGET_POS[0] + dx, self.TARGET_POS[1] + dy, self.TARGET_POS[2] + dz]
        else:
            pos = self.initial_pos
        vel = [self._rng.random() * (upper - lower) + lower for (lower, upper) in self.INITIAL_VEL_RANGE]
        ang_vel = [self._rng.random() * (upper - lower) + lower for (lower, upper) in self.INITIAL_ANG_VEL_RANGE]
        return pos, rpy, vel, ang_vel

    def _housekeeping(self):
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        p.loadURDF('sphere2.urdf', [0, 0, 2], globalScaling=0.1, physicsClientId=self.CLIENT, useFixedBase=True)
        pos, rpy, vel, ang_vel = self._sample_initial_state()

        self.DRONE_IDS = np.array(
            [p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + self.URDF),
                        pos,
                        p.getQuaternionFromEuler(rpy),
                        flags=p.URDF_USE_INERTIA_FROM_FILE,
                        physicsClientId=self.CLIENT
                        ) for i in range(self.NUM_DRONES)])

        for drone_id in self.DRONE_IDS:
            p.resetBaseVelocity(drone_id, vel, ang_vel, physicsClientId=self.CLIENT)

        #### Remove default damping #################################
        # for i in range(self.NUM_DRONES):
        #     p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        #### Show the frame of reference of the drone, note that ###
        #### It severly slows down the GUI #########################
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        #### Disable collisions between drones' and the ground plane
        #### E.g., to start a drone at [0,0,0] #####################
        # for i in range(self.NUM_DRONES):
        # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        ang_vel = state[13:16]
        vel = state[10:13]
        rot = R.from_quat(quat)
        a = rot.as_rotvec()

        state = self._getDroneStateVector(0)
        cost = (np.linalg.norm(pos - self.TARGET_POS)
                + 0.3 * (np.linalg.norm(a)
                        + np.linalg.norm(ang_vel)
                        + np.linalg.norm(vel))
                / np.linalg.norm(pos - self.TARGET_POS))
        return max(0, 2-cost)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(state[0:3] - self.TARGET_POS) < .0001:
            print("TERMINATED!")
            return True
        else:
            return False

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 3.5  # Truncate when the drone is too far away
                or abs(state[7]) > .4 or abs(state[8]) > .4  # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years
