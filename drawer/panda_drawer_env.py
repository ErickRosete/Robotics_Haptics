import gym
from gym import error, spaces, utils
from pathlib import Path
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

class PandaDrawerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, front=False, open=True):
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2]) 

        if front:
            self.rest_joints = [-9.21e-06, 0.37, 1.15e-05, -1.29, -4.12e-06, 1.70, 2.35, 0.0, 0.0]
        else:
            self.rest_joints = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08] # Initial joint angles
        
        if open:
            self.rest_joints.extend([0.04, 0.04])
        else:
            self.rest_joints.extend([0.0, 0.0])

        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*7), np.array([1]*7)) # x, y, z, joint_f1, joint_f2, force_f1, force_f2 

    def step(self, action, dt=0.005):
        """action: Velocities in xyz and fingers [vx, vy, vz, vf]"""

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi/2.]) # End effector desired orientation

        # Relative step in xyz
        dx = action[0] * dt
        dy = action[1] * dt
        dz = action[2] * dt
        df = action[3] * dt

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

        state_object = p.getLinkState(self.objectUid, 3)[0] # Handle grip Pose
        state_robot = p.getLinkState(self.pandaUid, 11, computeForwardKinematics=1)[0] # End effector pose xyz
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0]) # Fingers angle


        # Force measurements
        force = (self.getForce(9), self.getForce(10)) 

        # Reward calculation
        if state_object[0] <= 0.30: # Coord 'x' of object
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        info = {"object_position": state_object} 
        observation = np.asarray(state_robot + state_fingers + force)
        return observation, reward, done, info

    def getForce(self, linkIndex):
        contact_points = p.getContactPoints(bodyA=self.pandaUid, linkIndexA=linkIndex) 
        
        max_force, force = 1, 0
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
        urdfRootPath=pybullet_data.getDataPath()
        planeUid = p.loadURDF("plane.urdf", basePosition=[0,0,-0.65])

        # Load Panda robot

        self.pandaUid = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for i in range(len(self.rest_joints)):
            p.resetJointState(self.pandaUid, i, self.rest_joints[i]) 

        # Load objects
        tableUid = p.loadURDF("table/table.urdf",basePosition=[0.5, 0, -0.65])
        # trayUid = p.loadURDF("tray/traybox.urdf",basePosition=[0.65, 0, 0])

        # Load drawer
        p.setAdditionalSearchPath( str((Path.cwd() / "urdf").resolve()) )
        state_object= [0.90, random.uniform(-0.20, 0.20), 0.15]
        state_orientation = p.getQuaternionFromEuler((0, 0, np.pi))
        self.objectUid = p.loadURDF("drawer_one_sided_handle.urdf", 
                        basePosition=state_object, baseOrientation=state_orientation,
                        useFixedBase=True)
        p.setJointMotorControl2(self.objectUid, 0, p.VELOCITY_CONTROL, force=0.0) # Quit motor
        collisionFilterGroup = 17 #Intersect with raytest and default (1001)
        p.setCollisionFilterGroupMask(self.objectUid, -1, collisionFilterGroup, -1)

        # Get observation
        state_robot = p.getLinkState(self.pandaUid, 11, computeForwardKinematics=1)[0]
        print(state_robot)
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        force = (self.getForce(9), self.getForce(10)) 
        observation = np.asarray(state_robot + state_fingers + force)
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # rendering's back on again
        
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
