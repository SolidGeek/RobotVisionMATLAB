classdef Robot
    
    properties
        roboarm;
        robotool;
        robogripper;
    end
    
    methods
        function setXYZ(this,x,y,z)
            newPos = transl(x,y,z);
            this.roboarm.MoveJ(newPos);
        end
        
        function moveXYZ(this,x,y,z)
            newPos = this.roboarm.Pose() * transl(x,y,z);
            this.roboarm.MoveJ(newPos);
        end
        
        function moveX(this,x)
            newPos = this.roboarm.Pose() * transl(x,0,0);
            this.roboarm.MoveJ(newPos);
        end
        
        function moveY(this,y)
            newPos = this.roboarm.Pose() * transl(0,y,0);
            this.roboarm.MoveJ(newPos);
        end
        
        function moveZ(this,z)
            newPos = this.roboarm.Pose() * transl(0,0,z);
            this.roboarm.MoveJ(newPos);
        end
        
        function attach(this)
            attached = this.robotool.AttachClosest();
            if attached.Valid()
                this.robogripper.setJoints(32)
                attachedname = attached.Name();
                fprintf('Attached: %s\n', attachedname);
            else
                fprintf('No object is close enough\n');
            end
        end
        
        function detach(this)
                this.robogripper.setJoints(85)
                this.robotool.DetachAll();
                fprintf('Object is detached\n');
        end
        
        function open(this)
                this.robogripper.setJoints(85)
        end
        
        function close(this)
                this.robogripper.setJoints(32)
        end
        
    end
end