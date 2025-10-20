classdef UtilsFunctions

    
    % Methods
    methods

        %% constructor
        function obj = UtilsFunctions()
            % disp('Utility functions imported')
        end

    %% function for adding measurement noise
        function current_meas = func_gen_noise(current_meas,std,rot_flag)

            current_meas(1:6) = current_meas(1:6)+std*randn(6,1);
            
            if rot_flag
                R=reshape(current_meas(7:15),3,3);
                eul=rotm2eul(R);
                eul(1)=eul(1)+std*randn();
                eul(2)=eul(2)+std*randn();
                eul(3)=eul(3)+std*randn();
                Rnoisy=eul2rotm(eul);
                current_meas(7:15)=Rnoisy(:);
            end
        
        end

        %% Weightings for LMPC
        function [Q_x,R_u] = fcn_construct_weightings(m,n)

            nstates = 9*m+9*n; 
            nctrl = 9*m+9*n-17;
            % initialize the wigths
            Q_x = zeros(nstates);
            R_u = 0.1*eye(nctrl);


            Q_x(1:3,1:3) = 1e3*eye(3);
            Q_x(3*m+1:3*m+3,3*m+1:3*m+3) = 1e3*eye(3);
            Q_x(9*m+1:9*m+18,9*m+1:9*m+18) = 1e3*eye(18);
            % 

        end


        %% method for vector to skew-sym matrix (3x1 vector only)
        function S = skew(obj,v)
            if isvec(v,3)
                S = [  0   -v(3)  v(2)
                      v(3)  0    -v(1)
                     -v(2) v(1)   0];
            else
                error('error in skew function');
            end
        end

        %% function for vee operation (skew-sym matrix to vector)

        function vec = fcn_vee(obj,ss)
            
            if isa(ss,'sym')
                ss = expand(simplify(ss));
            end
            
            switch numel(ss)
                
                case 4
                    
                    if ~isequal(ss(1,2), -ss(2,1))
            %             warning('The provided matrix is not skew symmetric')
            %             disp(ss)
                    end
                    
                    vec = ss(1,2);
            
                case 9
            
                    if ~isequal(ss(3,2),-ss(2,3)) || ~isequal(ss(1,3),-ss(3,1)) ||....
                            ~isequal(ss(2,1),-ss(1,2))
            %             warning('The provided matrix is not skew symmetric.')
            %             disp(ss)
                    end
            
                    vec = [ss(3,2); ss(1,3); ss(2,1)];
            end
            
            if isa(ss,'sym')
                vec = simplify(vec);
            end
        end
 



        
    end
end