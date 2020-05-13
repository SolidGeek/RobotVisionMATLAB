%% RobotVision Camera sutff

% Link MATLAB with RoboDK
RDK = Robolink;

% Define the robot
robot = RDK.Item('KUKA KR 6 R700 sixx');
fprintf('Robot selected: %s\n', robot.Name());
%robot.setVisible(1);

disp(RDK.ItemList());

RDK.Cam2D_Close(0);

camera = RDK.Item('Camera');
camera_id = RDK.Cam2D_Add(camera);

RDK.Cam2D_Snapshot('test.jpg', camera_id);


% Now some python script, to do the image detection.. OPENCV BOIS

import py.find_lego_bricks.*