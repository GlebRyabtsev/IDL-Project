import random
import time

import numpy as np
import pkg_resources
import pybullet as p
import pybullet_data

from gym_pybullet_drones.envs import BaseRLAviary
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
                 act: ActionType = ActionType.RPM
                 ):
        self.Kp = np.array((-0.2, -0.2, -0.04))
        self.Kd = np.array((-0.06, -0.06, -0.01))
        self.INITIAL_POS_RANGE = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        self.INITIAL_ANG_RANGE = ((-0.5, 0.5), (-0.5, 0.5), (-np.pi / 2, np.pi / 2))
        self.INITIAL_VEL_RANGE = ((-1.0, 1.0),
                                  (-1.0, 1.0),
                                  (-1.0, 1.0))
        self.INITIAL_ANG_VEL_RANGE = ((-1.0, 1.0),
                                      (-1.0, 1.0),
                                      (-1.0, 1.0))

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def _preprocessAction(self, action):
        self.action_buffer.append(action)
        for k in range(action.shape[0]):  # not really needed since we're using one drone
            # ignoring act type
            state = self._getDroneStateVector()
            quat = state[3:7]
            ang_vel = state[13:16]
            rot = R.from_quat(quat)
            rotvec = R.from_quat(quat).as_rotvec()
            torque = self.Kp * rot.apply(rotvec, inverse=True) + self.Kd * rot.apply(ang_vel, inverse=True)

    def __sample_initial_state(self, seed):
        rng = np.random.default_rng(seed)
        rpy = [rng.random() * (upper - lower) + lower for (lower, upper) in self.INITIAL_ANG_RANGE]
        pos = [rng.random() * (upper - lower) + lower for (lower, upper) in self.INITIAL_POS_RANGE]
        vel = [rng.random() * (upper - lower) + lower for (lower, upper) in self.INITIAL_VEL_RANGE]
        ang_vel = [rng.random() * (upper - lower) + lower for (lower, upper) in self.INITIAL_ANG_VEL_RANGE]
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

        self.DRONE_IDS = np.array(
            [p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + self.URDF),
                        self.INIT_XYZS[i, :],
                        p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                        flags=p.URDF_USE_INERTIA_FROM_FILE,
                        physicsClientId=self.CLIENT
                        ) for i in range(self.NUM_DRONES)])
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
