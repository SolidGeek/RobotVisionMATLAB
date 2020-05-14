%% RobotVision Mini-Project

clc;

addpath('functions');

% Link MATLAB with RoboDK
RDK = Robolink;
robot = Robot;

% Display a list of all items within the RoboDK Environment
fprintf('Available items in the station:\n');
disp(RDK.ItemList());

world_frame = RDK.Item('World Frame');
fprintf('World Frame: \t%s\n', world_frame.Name());

% Define the robot
roboArm = RDK.Item('KUKA KR 6 R700 sixx');
fprintf('Robot selected: \t%s\n', roboArm.Name());
robot.roboarm = roboArm; 

ref_base = roboArm.Parent();
fprintf('Robot base frame: \t%s\n', ref_base.Name());

object = RDK.Item('Plate');
fprintf('Object selected: \t%s\n', object.Name());

ref_object = RDK.Item('LEGO Bricks Frame');
fprintf('Reference frame: \t%s\n', ref_object.Name());

tool = RDK.Item('Gripper');
fprintf('Tool selected: \t\t%s\n', tool.Name());
robot.robotool = tool;

targetHome = RDK.Item('Home');
fprintf('Home Position: \t\t%s\n', targetHome.Name());

jhome = targetHome.Joints();
RDK.setSimulationSpeed(5);

roboArm.setPoseFrame(world_frame);

roboArm.MoveJ(jhome); % Joint move


robot.setXYZ(515.875,-181.5,5);

robot.attach()

robot.setXYZ(523.784,321.229,5);

robot.detach()

roboArm.MoveJ(jhome); % Joint move

robot.setXYZ(523.784,321.229,5);

robot.attach()

robot.setXYZ(515.875,-181.5,5);

robot.detach()

roboArm.MoveJ(jhome); % Joint move
