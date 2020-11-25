import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import gym
from gym import error, spaces, utils, ObservationWrapper, ActionWrapper, RewardWrapper
from pathlib import Path
import pybullet as p
import pybullet_data
import math
import numpy as np
from utils.utils import get_cwd
import logging

class PandaPegEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, show_gui=False, dt=0.005):
        if show_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.logger = logging.getLogger(__name__)

        self.data_dir = str((get_cwd() / 'data').resolve())
        self.logger.info("Data Folder: %s" % self.data_dir)

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2]) 

        self.dt = dt
        self.initial_pose = (0.8, 0.0, 0.5)
        self.initial_orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi/2.])
        
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*7), np.array([1]*7)) # x, y, z, joint_f1, joint_f2, force_f1, force_f2 
        

    def step(self, action, dt=0.005):
        """action: Velocities in xyz and fingers [vx, vy, vz, vf]"""
        # self.dt = dt
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi/2.]) # End effector desired orientation

        # Relative step in xyz
        dx = action[0] * self.dt
        dy = action[1] * self.dt
        dz = action[2] * self.dt
        df = action[3] * self.dt

        currentPose = p.getLinkState(self.pandaUid, 11, computeForwardKinematics=1) # End effector pose
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, newPosition, orientation)

        fingerPose = p.getJointState(self.pandaUid, 9)[0]
        fingerAngle = fingerPose + df

        p.setJointMotorControlArray(self.pandaUid, 
                                    list(range(7))+[9,10], #7 DoF + 2 fingers
                                    p.POSITION_CONTROL, 
                                    list(jointPoses[:7])+2*[fingerAngle]) # Desired position
        p.stepSimulation()

        state_target = p.getLinkState(self.boardUid, self.target)[0]
            # p.addUserDebugText(str(i), state_object)

        #Observation
        state_robot = p.getLinkState(self.pandaUid, 11, computeForwardKinematics=1)[0] # End effector pose xyz
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0]) # Fingers angle
        force = (self.getForce(9), self.getForce(10)) 
        observation = np.asarray(state_robot + state_fingers + force)

        done, success  = self.getTermination()
        if done:
            if success:
                reward = 1000       # High positive reward to incentivize learning with success
            else:
                reward = -1000      # High negative reward to avoid learning termination without success
        else:
            reward = self.getReward()   # Continuous reward    

        info = {"target_position": state_target, "success": success} 
        return observation, reward, done, info

    def getReward(self):
        
        peg_pose = np.asarray(p.getBasePositionAndOrientation(self.pegUid)[0])
        target_pose = np.asarray(p.getLinkState(self.boardUid, self.target)[0])

        offset = 0.1 # 20 cm on top
        target_pose[2] += offset
        
        # Delete offset
        tolerance = 0.05
        if (target_pose[0] - tolerance < peg_pose[0] < target_pose[0] + tolerance and # coord 'x' and 'y' of object
            target_pose[1] - tolerance < peg_pose[1] < target_pose[1] + tolerance and 
            target_pose[2] - tolerance < peg_pose[2] < target_pose[2] + tolerance): # Coord 'z' of object
            target_pose[2] -= offset

        dist = np.linalg.norm(target_pose - peg_pose)
        reward = -dist  # Minimize dist
        return reward

    def getTermination(self):
        done, success = False, False
        peg_pose = np.asarray(p.getBasePositionAndOrientation(self.pegUid)[0])
        board_pose = np.asarray(p.getBasePositionAndOrientation(self.boardUid)[0])
        state_robot = np.asarray(p.getLinkState(self.pandaUid, 11, computeForwardKinematics=1)[0]) # End effector pose xyz

        if (board_pose[0] - 0.05 < peg_pose[0] < board_pose[0] + 0.05 and # coord 'x' and 'y' of object
            board_pose[1] - 0.05 < peg_pose[1] < board_pose[1] + 0.05 and 
            peg_pose[2] <= 0.20): # Coord 'z' of object
            # Inside box
            done, success = True, True
        elif np.linalg.norm(state_robot - peg_pose) > 0.075:
            # Peg dropped outside box
            done = True
        
        return done, success

    def getForce(self, linkIndex):
        contact_points = p.getContactPoints(bodyA=self.pandaUid, linkIndexA=linkIndex) 
        
        max_force, force = 100, 0
        if len(contact_points) > 0:
            for cp in contact_points:
                force += cp[9]
            if force > max_force:
                force = max_force
            force /= max_force
        return force


    def getDepth(self, linkIndex):
        if linkIndex != 9 and linkIndex !=10:
            return

        w, h = 48, 48
        com_p, com_o, _, _, _, _ = p.getLinkState(self.pandaUid, linkIndex, computeForwardKinematics=1)
        rot_matrix = np.array(p.getMatrixFromQuaternion(com_o)).reshape(3, 3)

        # Initial vectors
        down_vector = (0, 0, 1)
        if linkIndex == 9:
            ray_dir = (0, -1, 0) # y-axis
        elif linkIndex == 10:
            ray_dir = (0, 1, 0) # y-axis
        com_p = com_p + rot_matrix @ down_vector * 0.025   # offset to pose of contact area

        # Transformations to obtain rayFrom and rayTo
        axis1 = np.linspace(-0.015, 0.015, w)
        axis2 = np.linspace(-0.015, 0.015, h)
        plane1, plane2 = np.meshgrid(axis1, axis2)
        rayFrom = np.stack((plane1, np.zeros((w, h)), plane2), axis=-1) #Plane
        rayFrom = rayFrom @ rot_matrix #Rotate matrix
        rayFrom += com_p #Translate points
        rayTo = rayFrom + rot_matrix @ ray_dir * 0.015
        rayFrom = rayFrom.reshape(-1,3).tolist()
        rayTo = rayTo.reshape(-1,3).tolist()

        info = p.rayTestBatch(rayFrom, rayTo, numThreads=0, collisionFilterMask=16) #Only measure collisions with group 16 (1000)
        depth = []
        for obj, li, fr, pos, norm in info:
            depth.append(fr)
        depth = np.array(depth).reshape(w,h)

        for i in range(len(rayFrom)):
            p.addUserDebugLine(rayFrom[i], rayTo[i])

        return depth

    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        p.setGravity(0,0,-10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())


        # Loading objects in environment
        planeUid = p.loadURDF("plane.urdf", basePosition=[0,0,-0.65])

        # Load Panda robot
        self.pandaUid = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        rest_joints = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08, 0.024, 0.024] # Initial joint angles
        for i in range(len(rest_joints)):
            p.resetJointState(self.pandaUid, i,  rest_joints[i]) 

        # Load objects
        tableUid = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])

        # Load Peg in hole
        p.setAdditionalSearchPath(self.data_dir)
        board_pose= [np.random.uniform(0.60, 0.70), np.random.uniform(-0.2, 0.2), 0.0]
        board_orientation = p.getQuaternionFromEuler((0, 0, -np.pi/2))
        self.boardUid = p.loadURDF("peg_in_hole_board/peg_in_hole_board.urdf", basePosition=board_pose, baseOrientation=board_orientation, \
                                    useFixedBase=True)

        self.target = np.random.randint(low=0, high=3)
        peg_position = list(p.getLinkState(self.pandaUid, 11, computeForwardKinematics=1)[0])
        peg_position[2] -= 0.02
        if self.target == 0:
            self.pegUid = p.loadURDF("cylinder/cylinder.urdf", basePosition=peg_position,
                                baseOrientation=board_orientation)
        elif self.target == 1:
            self.pegUid = p.loadURDF("hexagonal_prism/hexagonal_prism.urdf", basePosition=peg_position,
                                baseOrientation=board_orientation)
        elif self.target == 2:
            self.pegUid = p.loadURDF("square_prism/square_prism.urdf", basePosition=peg_position,
                                baseOrientation=board_orientation)

        collisionFilterGroup = 17 #Intersect with raytest and default (1001)
        p.setCollisionFilterGroupMask(self.pegUid, -1, collisionFilterGroup, -1)

        # Get observation
        state_robot = p.getLinkState(self.pandaUid, 11, computeForwardKinematics=1)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        force = (self.getForce(9), self.getForce(10)) 
        observation = np.asarray(state_robot + state_fingers + force)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # rendering's back on again
        
        #Debugging
        p.addUserDebugText("G1", (0,0.5,0.5), textSize=0.75)
        target_pose = np.asarray(p.getLinkState(self.boardUid, self.target)[0])
        p.addUserDebugText("G2", target_pose, textSize=0.75)
        target_pose[2] += 0.1
        p.addUserDebugText("G3", target_pose, textSize=0.75)

        return observation

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                        distance=.7,
                                                        yaw=90,
                                                        pitch=-70,
                                                        roll=0,
                                                        upAxisIndex=2)

        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                aspect=float(960) /720,
                                                nearVal=0.1,
                                                farVal=100.0)

        (w, h, img, depth, segm) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img = np.asarray(img).reshape(720, 960, 4) # H, W, C
        rgb_array = img[:, :, :3] # Ignore alpha channel
        return rgb_array

    def close(self):
        p.disconnect()


# Custom wrappers
class TransformObservation(ObservationWrapper):
    def __init__(self, env=None, withForce = True, withJoint=False, relative = True, noise=False):
        super(TransformObservation, self).__init__(env)

        self.withForce = withForce
        self.withJoint = withJoint
        self.relative = relative
        self.noise = noise

        if withForce:
            if withJoint:
                self.observation_space = spaces.Box(np.array([-1]*7), np.array([1]*7)) # x, y, z, j1, j2, force_f1, force_f2 
            else:
                self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5)) # x, y, z, force_f1, force_f2 
        else: # mode = "pose"
            if withJoint:
                self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5)) # x, y, z, j1, j2 
            else:
                self.observation_space = spaces.Box(np.array([-1]*3), np.array([1]*3)) # x, y, z 

    def observation(self, obs):
        if self.relative:
            state_target = np.asarray(p.getLinkState(self.env.boardUid, self.env.target)[0])
            state_target[2] += 0.0175 # Target is a little bit on top
            if self.noise:
                state_target +=  np.random.normal(0, 0.01)
            obs[:3] = obs[:3] - state_target  # Relative pose to target
            obs[3:] = obs[3:] - np.array([0.024, 0.024, 0.6, 0.6]) # Relative Joints and Force to target

        if not self.withJoint:
            obs = obs[[0,1,2,5,6]]
        if not self.withForce:
            obs = obs[:-2]

        return obs

class TransformAction(ActionWrapper):
    def __init__(self, env=None, withJoint=False):
        super(TransformAction, self).__init__(env)
        self.withJoint = withJoint

        if withJoint:
            self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4)) # vel_x, vel_y, vel_z, vel_joint
        else:
            self.action_space = spaces.Box(np.array([-1]*3), np.array([1]*3)) # vel_x, vel_y, vel_z


    def action(self, action):
        if self.withJoint:
            return action
        else:
            action = np.append(action, -0.002/self.env.dt) 
            return action

class TransformReward(RewardWrapper):
    def __init__(self, env, sparse=False):
        super(TransformReward, self).__init__(env)
        self.env = env
        self.sparse = sparse

    def reward(self, rew):
        if self.sparse:
            done, success  = self.env.getTermination()
            if done:
                if success:
                    return 1.0
                else:
                    return -1.0
            else:
                return 0.0
        else:
            return rew


# Create environment with custom wrapper
def pandaPegV2(show_gui=False, dt=0.005, withForce=True, withJoint=False, relative = True, noise=False, sparse=False):
    env = PandaPegEnv(show_gui, dt)
    env = TransformObservation(env, withForce, withJoint, relative, noise)
    env = TransformAction(env, withJoint)
    env = TransformReward(env, sparse)
    return env

if __name__ == "__main__":
    env = pandaPegV2()
    for episode in range(100):
        s = env.reset()
        episode_length, episode_reward = 0,0
        for step in range(500):
            a = env.action_space.sample()
            s, r, done, _ = env.step(a)
            if step % 100 == 0:
                print("Action", a)
                print("State", s)
            if(done):
                break
