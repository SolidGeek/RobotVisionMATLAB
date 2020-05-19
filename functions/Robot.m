% Robot Class: 
% The robot class defines functions for moving and operating the robot
% and its tools

classdef Robot
    
    properties
        roboarm;        % Retrieve the robot object
        robotool;       % Retrieve the tool reference frame
        robogripper;    % Retrieve the RobotiQ tool for controls
    end
    
    methods
        
        % Function that move joints to target
        function MoveJ(this,target)
            this.roboarm.MoveJ(target);
        end
        
        function value = addTrans(this, x,y,z)
            value = [0 0 0 x; 0 0 0 y; 0 0 0 z; 0 0 0 0];
        end
        
        % Function that set X Y Z coordinates and move joints to position
        function setXYZ(this,x,y,z)
            newPos = transl(x,y,z);
            this.roboarm.MoveJ(newPos);
        end
        
        function setTrans(this, trans)
            this.roboarm.MoveJ(trans)
        end
        
        % Function move in X Y Z from current position
        function moveXYZ(this,x,y,z)
            newPos = this.roboarm.Pose() * transl(x,y,z);
            this.roboarm.MoveJ(newPos);
        end
        
        % Function move in X from current position
        function moveX(this,x)
            newPos = this.roboarm.Pose() * transl(x,0,0);
            this.roboarm.MoveJ(newPos);
        end
        
        % Function move in Y from current position
        function moveY(this,y)
            newPos = this.roboarm.Pose() * transl(0,y,0);
            this.roboarm.MoveJ(newPos);
        end
        
        % Function move in Z from current position
        function moveZ(this,z)
            newPos = this.roboarm.Pose() * transl(0,0,z);
            this.roboarm.MoveJ(newPos);
        end
        
        % Function that attaches object to the tool
        function attach(this)
            attached = this.robotool.AttachClosest();
            
            % Check if an object has been attached
            if attached.Valid()
                
                this.robogripper.setJoints(32) % Close the RobotiQ gripper
                attachedname = attached.Name();
                fprintf('Attached: %s\n', attachedname);
            else
                fprintf('No object is close enough\n');
            end
        end
        
        % Function that detaches the object from the tool 
        function detach(this)
                this.robogripper.setJoints(85) % Open the RobotiQ gripper
                this.robotool.DetachAll();
                fprintf('Object is detached\n');
        end
        
        % Function that opens the RobotiQ Gripper
        function open(this)
                this.robogripper.setJoints(85)
        end
        
        % Function that closes the RobotiQ Gripper
        function close(this)
                this.robogripper.setJoints(32)
        end
        
    end
end