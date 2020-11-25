import pybullet as p
from pathlib import Path
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)
p.resetSimulation()
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
p.setGravity(0,0,-10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Loading objects in environment
urdfRootPath=pybullet_data.getDataPath()
planeUid = p.loadURDF("plane.urdf", basePosition=[0,0,-0.65])

# Load Panda robot
pandaUid = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
rest_joints = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08, 0.023, 0.023] # Initial joint angles
for i in range(len(rest_joints)):
    p.resetJointState(pandaUid, i,  rest_joints[i]) 

# Load objects
tableUid = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])

# Load Peg in hole
p.setAdditionalSearchPath( str((Path.cwd() / "urdf").resolve()) )
state_object= [np.random.uniform(0.60, 0.90), np.random.uniform(-0.2, 0.2), 0.0]
state_orientation = p.getQuaternionFromEuler((0, 0, -np.pi/2))
boardUid = p.loadURDF("peg_in_hole_board.urdf", basePosition=state_object, baseOrientation=state_orientation)

target = np.random.randint(low=0, high=3)
peg_position = list(p.getLinkState(pandaUid, 11, computeForwardKinematics=1)[0])
peg_position[2] -= 0.005

if target == 0:
    pegUid = p.loadURDF("cylinder.urdf", basePosition=peg_position,
                         baseOrientation=state_orientation)
elif target == 1:
    pegUid = p.loadURDF("hexagonal_prism.urdf", basePosition=peg_position,
                         baseOrientation=state_orientation)
elif target == 2:
    pegUid = p.loadURDF("square_prism.urdf", basePosition=peg_position,
                         baseOrientation=state_orientation)

collisionFilterGroup = 17 #Intersect with raytest and default (1001)
p.setCollisionFilterGroupMask(pegUid, -1, collisionFilterGroup, -1)

# # Get observation
# state_robot = p.getLinkState(self.pandaUid, 11, computeForwardKinematics=1)[0]
# print(state_robot)
# state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
# force = (self.getForce(9), self.getForce(10)) 
# observation = np.asarray(state_robot + state_fingers + force)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # rendering's back on again

input("Press Enter to continue...")
