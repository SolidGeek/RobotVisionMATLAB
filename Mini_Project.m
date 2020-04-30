%% RobotVision Mini-Project

% Link MATLAB with RoboDK
RDK = Robolink;

% Set the path for the RoboDK project
path = RDK.getParam('../RobotVisionMATLAB');

% Add the specific RoboDK project to the MATLAB project
RDK.AddFile([path, 'robot_environment.rdk']);

% Display a list of all items within the RoboDK Environment
fprintf('Available items in the station:\n');
disp(RDK.ItemList());

% Define the robot
robot = RDK.Item('KUKA KR 6 R700 sixx');
fprintf('Robot selected: %s\n', robot.Name());
%robot.setVisible(1);


% Retrieve the base frame of the robot
ref_base = robot.Parent();
fprintf('Robot base frame: %s\n', ref_base.Name());

object = RDK.Item('Block');
fprintf('Object selected: %s\n', object.Name());

ref_object = object.Parent();
fprintf('Reference frame: %s\n', ref_object.Name());

tool = RDK.Item('Gripper RobotiQ 85 Opened');
fprintf('Tool selected: %s\n', tool.Name());

target1 = RDK.Item('Home');
fprintf('Target 1 selected:\t%s\n', target1.Name());

target2 = RDK.Item('Pickup');
fprintf('Target 2 selected:\t%s\n', target2.Name());


% Get the joint values for the first target (home target):
% jhome = [ 0, 0, 0, 0, 30, 0];
jhome = target1.Joints();
Pickup = target2.Joints();

% Set the simulation speed. This is a ratio, for example, simulation speed
% of 5 (default) means that 1 second of simulated time corresponds to 1
% second of real time.
RDK.setSimulationSpeed(1);

% Display the current joint values of the robot
fprintf('Current robot joints:\n');
joints = robot.Joints();
disp(joints);

% Set the robot at the home position
robot.setJoints(jhome); % Immediate move

fprintf('Moving to Pickup\n');
robot.MoveJ(Pickup)

fprintf('Done!\n');







