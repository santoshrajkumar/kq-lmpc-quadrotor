classdef KoopmanLift
    
    % Properties
    properties
        mass % mass
        J % inertia
        d % distance from the center of mass to the center of each rotor
        M % # of observables for pos part
        N % # of observables for SO(3) part
        g % acceleration due to gravity
        utilsf % utility functions/methods
    end
    
    % Methods
    methods
        %% Constructor
        function obj = KoopmanLift(params,M,N)

            % import the utility methods
            import UtilsFunctions.*

            % Point2D constructor
            if nargin == 3
                obj.mass = params.m;
                obj.J = params.J;
                obj.d = params.d;
                obj.M=M;
                obj.N=N;
                obj.g=params.g;
                obj.utilsf=UtilsFunctions();
            else
                disp('Invalid entry of in KoopLift')
            end
        end

        %% Method to convert actual states (18 #) to lifted states in se3
        function X_out= fcn_gen_koopman_states_se3(obj,Xin)
             % Xin is the 18x1 vector of the actual states
             % Xin=[x v R(:) omega]'
             % extract the components
             R=reshape(Xin(7:15),3,3); x=Xin(1:3); v=Xin(4:6);w=Xin(16:18);
             
             gvec=[0;0;obj.g]; % g*e3
             wx=obj.utilsf.skew(w); %skewed omega
             
             % initialize the lifted states h, y, p
             hs=[]; ys=[];ps=[];
             % populate h, y, p vectors
             for i=1:obj.M
                hs=[hs; -((wx')^(i-1))*R'*gvec];
                ys =[ys; ((wx')^(i-1))*R'*v];
                ps =[ps; ((wx')^(i-1))*R'*x];
             end
                
             %Koopman linear state Xout=[p y h z]'
             X_out = [ps;ys;hs; obj.fcn_gen_koopman_states_so3(R,w)];
        
        end

        %% method to convert SO(3) actual states to SO(3) lifted states
        function X = fcn_gen_koopman_states_so3(obj,R,w)
            
            n=obj.N;

            dimR = size(R);
            
            if dimR(2) == 1
                R = reshape(R,3,3);
            end
            
            X = zeros(9*n,1);
            wx=obj.utilsf.skew(w);
            
            for k=1:n
                z_nvec=R*(wx)^(k-1); % zk = R(wx)^(k-1)
                X(9*k-8:9*k)=z_nvec(:); % reshape and assign in the X vector
            end
        
        end

        %% Method to convert lifted states in SE(3) to the actaul states (18 #)
        function X_out= fcn_se3_states_to_actual(obj,Xin)
             % Xin has the structure [p y h z]

             % z_1 is the R(:)
             R=reshape(Xin(9*obj.M+1:9*(obj.M+1)),3,3);
             % rehsape(z_2,3,3) is R*skew(omega)
             % skew(omega)=R'*rehsape(z_2,3,3)
             omega_x = R'*reshape(Xin(9*(obj.M+1)+1:9*(obj.M+1)+9),3,3);
             % p_1 = R'*x => x = R*p_1
             x = R*Xin(1:3);
             % y_1 = R'*v => v=R*y_1
             v=R*Xin(3*obj.M+1:3*(obj.M+1));
             % actual state Xout = x v R(:) omega]'
             X_out = [x;v;R(:);obj.utilsf.fcn_vee(omega_x)];
        end

        %% Method to Construct the lifted A Matrix
        function A  = fcn_A_lifted(obj)
            m=obj.M; n=obj.N;

            A= zeros(9*m+9*n);
            Ap = obj.fcn_lifted_A_position_dynamics();
            Aa=obj.fcn_construct_A_so3();
            A(1:9*m,1:9*m)=Ap;
            A(9*m+1:end,9*m+1:end)=Aa;
        end
        
        %% Method to Construct the lifted Ap Matrix

        function Ap  = fcn_lifted_A_position_dynamics(obj)
            
            m=obj.M; 

            Ap=zeros(9*m);
            
            Ap(1:3*(m-1),4:3*m)= eye(3*(m-1));
            Ap(1:3*(m-1),3*m+1:6*m-3)=eye(3*(m-1));

            Ap(3*m+1:6*m-3,3*m+4:6*m) = eye(3*(m-1));
            Ap(3*m+1:6*m-3,6*m+1:9*m-3) = eye(3*(m-1));

            Ap(6*m+1:9*m-3,6*m+4:9*m) = eye(3*(m-1));

        end
        
        %% Method to Construct the lifted A Matrix for the SO(3) dynamics

        function A_so3=fcn_construct_A_so3(obj)
            
            A_so3 = zeros(9*obj.N,9*obj.N);
    
            for i=1:obj.N
                if i < obj.N
                    A_so3(9*i-8:9*i,9*i+1:9*i+9) = eye(9);
                end
            end

        end
        %% Method to construct the constant B (Bbar) matrix
        
        function [Bbar,Bp,Ba] = fcn_Bbar(obj)
            m=obj.M; n=obj.N;
            % initialize the Bbar matrix
            Bbar = zeros(9*m+9*n,9*m+9*n-17);
            
            % initialize Bp & Ba matrices
            Bp = zeros(9*m,9*m-8);
            Ba = zeros(9*n,9*(n-1));
            
            % construct the Bp matrix
            Bp(4:3*m,1:3*(m-1)) = eye(3*(m-1));
            Bp(3*m+3,3*m-2)=1;
            Bp(3*m+4:6*m,3*m-1:6*m-5) = eye(3*(m-1));
            Bp(6*m+4:9*m,6*m-4:9*m-8) = eye(3*(m-1));
            % construct the Ba matrix
            Ba(10:end,:) = eye(9*(n-1));

            % assign Bbar & Bp matrices
            Bbar(1:9*m,1:9*m-8) = Bp;
            Bbar(9*m+1:end,9*m-7:end) = Ba;
 
        end

        %% Method to construct the state dependent B matrix in SE(3)

        function calB=fcn_CalB(obj,x_nl,x_l)
            m=obj.M; n=obj.N;

            [~,~,Bbar_so3]= obj.fcn_Bbar();
            p1=x_l(1:3);
            y1=x_l(3*m+1:3*m+3);
            h1=x_l(6*m+1:6*m+3);
            
            calBpos = obj.fcn_construct_calBpos(x_nl(16:18),h1,y1,p1);
            
            calB=zeros(9*m+9*n,4);
            
            calB_so3 = obj.fcn_construct_calB_so3(reshape(x_nl(7:15),3,3),....
                                           obj.utilsf.skew(x_nl(16:18)),Bbar_so3);
            
            calB(1:9*m,:)=calBpos;
            calB(9*m+1:end,2:end)=calB_so3;

        end

        %% Method to construct part of calB involving the position dynamics

        function calBpos= fcn_construct_calBpos(obj,w,h1,y1,p1)
            m=obj.M;

            calBpos=zeros(9*m,4);
            e3=[0;0;1];
            Omega=obj.utilsf.skew(w);
            OmegaT=Omega';
            
            Hk=obj.fcn_gen_H_Y_P(w,h1);
            Yk=obj.fcn_gen_H_Y_P(w,y1);
            Pk=obj.fcn_gen_H_Y_P(w,p1);
            
            % for i=1:m
            %     calY(:,:,i) = (1/params.m)*(OmegaT^(i-1))*e3;
            % end
            
            for j=1:m
                calY = (1/obj.mass)*(OmegaT^(j-1))*e3;
                calBpos(3*j-2:3*j,2:end)=Hk(:,:,j);
                calBpos(3*m+3*j-2:3*m+3*j,1) = calY;
                calBpos(3*m+3*j-2:3*m+3*j,2:end)=Yk(:,:,j);
                calBpos(6*m+3*j-2:6*m+3*j,2:end)=Pk(:,:,j);
            end

        end

        %% Helper function for constructing the part of calB involving the position dynamics

        function HYP=fcn_gen_H_Y_P(obj,w,hyp1)

            Omega=obj.utilsf.skew(w);
            OmegaT=Omega';
            m=obj.M;

            for k=2:m
                temp=zeros(3,3);
                for i=1:(k-1)
                    temp = temp+(OmegaT^(i-1))*....
                            (obj.utilsf.skew((OmegaT^(k-1-i))*hyp1))*inv(obj.J);
                end
               
                HYP(:,:,k)=temp;
            end

        end

        %% Method to construct the calB (state dependent) for the SO(3) part

        function calB = fcn_construct_calB_so3(obj,R,wx,Bbar)

            invJ = inv(obj.J);
            n=obj.N;

            Bh=zeros(9*(n-1),3);
        
            % construct the u vector
            for ii=1:(n-1)
            
                c_ii1_nvec = zeros(3,3);
                c_ii2_nvec = zeros(3,3);
                c_ii3_nvec = zeros(3,3);
                   for jj=1:ii
                        c_ii1_nvec =  c_ii1_nvec+R*wx^(jj-1)*obj.utilsf.skew(invJ(:,1))*wx^(ii-jj);
                        c_ii2_nvec =  c_ii2_nvec+R*wx^(jj-1)*obj.utilsf.skew(invJ(:,2))*wx^(ii-jj);
                        c_ii3_nvec =  c_ii3_nvec+R*wx^(jj-1)*obj.utilsf.skew(invJ(:,3))*wx^(ii-jj);
                   end
                C_ii = [c_ii1_nvec(:) c_ii2_nvec(:) c_ii3_nvec(:)];
            
                % u_ii=C_ii*ubar;
            
                Bh(9*ii-8:9*ii,:)=C_ii;
            end
        
            calB=Bbar*Bh;

        end


        
    end
end