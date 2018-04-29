%% VMC_SOM_trial_4 Updated on 27-Apr-2018
% upon training work space is stored in
% VMC_data_27_Apr_9_36_AM_10000_samples.mat file
clear
close all
clc

%% Init
ws_x = 0.2;     %workspace length,breadth & height
ws_y = 0.2;
ws_z = 0.2;
samples = 10000;
som_x = 12;
som_y = 7;
som_z = 4;
sigma_i = 2;        %initial sigma
sigma_f = 0.1;      %final sigma
sigma_dash_i = 2.5;
sigma_dash_f = 0.01;
eps_i = 1;
eps_f = 0.05;
eps_dash_i = 0.9;
eps_dash_f = 0.9;

mn_sqr_err = zeros(samples,1);

%% Generation of data for input space
input_space = [ws_x*rand(1,samples)-0.1;ws_y*rand(1,samples)+0.2;ws_z*rand(1,samples)+0.4];
% scatter3(input_space(1,:),input_space(2,:),input_space(3,:),'.','linewidth',1)
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% hold on

%% Initialization of SOM
weights = 0.15*rand(som_x,som_y,som_z,3);   % 3 inputs
Theta = rand(som_x,som_y,som_z,3);          % 3 outputs
A = rand(som_x,som_y,som_z,3,3);            % dimensions 3x3

%% SOM Training Algo
for i = 1:samples
    i
    sigma = sigma_i*((sigma_f/sigma_i)^(i/samples));
    sigma_dash = sigma_dash_i*((sigma_dash_f/sigma_dash_i)^(i/samples));
    eps = eps_i*((eps_f/eps_i)^(i/samples));
    eps_dash = eps_dash_i*((eps_dash_f/eps_dash_i)^(i/samples));
    
    %For each training step, a target location is presented at randomly
    %chosen location within the workspace of the robot, hence....
    
    % Finding out winning Neuron...........................................
    dist = zeros(som_x,som_y,som_z);     %distances
    for x = 1:som_x
        for y = 1:som_y
            for z = 1:som_z
                w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
                dist(x,y,z) = norm(w-input_space(:,i));
            end
        end
    end
    [C,I] = min(dist(:));
    [I1,I2,I3] = ind2sub(size(dist),I);
    index_min = [I1;I2;I3]; %index of winning neuron
    
    % Finding output corresponding to current sample i.e current input.....
    Theta_0_out = zeros(3,1);
    h = zeros(som_x,som_y,som_z);        %neighbourhood functions
    h_total = 0;                         %sum of all neighbourhood functions 's'
    for x = 1:som_x
        for y = 1:som_y
            for z = 1:som_z
                w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
                eucl_dist = norm([x;y;z]-index_min);
                h(x,y,z) = exp(-(eucl_dist^2/(2*sigma^2)));
                h_total = h_total + h(x,y,z);
                
                Theta_lambda = [Theta(x,y,z,1);Theta(x,y,z,2);Theta(x,y,z,3)];
                
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                
                W_lambda = w;    
                % Theta_0_out for coarse action
                Theta_0_out = Theta_0_out + h(x,y,z)*(Theta_lambda + A_lambda*(input_space(:,i)-w));
            end
        end
    end
    Theta_0_out = Theta_0_out/h_total;
    
    %Moving the Manipulator................................................
    %This is the initial output which causes the coarse movement of the
    %endeffector to reach a position having coordinates as V0.
    %Coarse Action.........................................................
    V0 = fwd_kin(Theta_0_out);
    
    %Fine Action...........................................................
    Theta_temp = 0;
    for x = 1:som_x %hence calculating in this for loop only
        for y = 1:som_y
            for z = 1:som_z              
                eucl_dist = norm([x;y;z]-index_min);        
              
                         
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                   
                % Theta_0_out for coarse action
                Theta_temp = Theta_temp + h(x,y,z)*(A_lambda*(input_space(:,i)-V0));               
              
            end
        end
    end
    Theta_1_out = Theta_0_out + Theta_temp/h_total;
    
    V1 = fwd_kin(Theta_1_out);
    
    %Updation..............................................................
    delta_v = V1-V0;
    delta_theta = Theta_1_out - Theta_0_out;
    
    %pre calculation of factors required for delta_theta_lambda and delta_A
    pre_fac_theta = 0;    %This is the factor we will require for delta_theta_lambda
    pre_fac_A = 0;        %and delta_A hence calculating in this for loop

    for x = 1:som_x 
        for y = 1:som_y
            for z = 1:som_z
                
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                
                Theta_lambda = [Theta(x,y,z,1);Theta(x,y,z,2);Theta(x,y,z,3)];
                
                W_lambda = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
                
                %Prefactor calculation for delta_theta_lambda and delta_A
                pre_fac_theta = pre_fac_theta + h(x,y,z)*(Theta_lambda + A_lambda*(V0-W_lambda));
                pre_fac_A = pre_fac_A + h(x,y,z)*A_lambda*delta_v;
            end
        end
    end
    pre_fac_theta = Theta_0_out - pre_fac_theta/h_total;
    pre_fac_A = delta_theta - pre_fac_A/h_total;
    
    for x = 1:som_x 
        for y = 1:som_y
            for z = 1:som_z
                delta_Theta_lambda = (h(x,y,z)/h_total)*pre_fac_theta;
                delta_A = (h(x,y,z)/(h_total*norm(delta_v)^2))*pre_fac_A*delta_v';
                
                % w - update...............................................
                weights(x,y,z,1) = weights(x,y,z,1) + eps*h(x,y,z)*(input_space(1,i)-weights(x,y,z,1));
                weights(x,y,z,2) = weights(x,y,z,2) + eps*h(x,y,z)*(input_space(2,i)-weights(x,y,z,2));
                weights(x,y,z,3) = weights(x,y,z,3) + eps*h(x,y,z)*(input_space(3,i)-weights(x,y,z,3));
                
                % Theta - update...........................................
                Theta(x,y,z,1) = Theta(x,y,z,1) + eps_dash*delta_Theta_lambda(1);
                Theta(x,y,z,2) = Theta(x,y,z,2) + eps_dash*delta_Theta_lambda(2);
                Theta(x,y,z,3) = Theta(x,y,z,3) + eps_dash*delta_Theta_lambda(3);
                
                % A - update...............................................
                for A_row = 1:3
                    for A_col = 1:3
                        A(x,y,z,A_row,A_col) = A(x,y,z,A_row,A_col) + eps_dash*delta_A(A_row,A_col);
                    end
                end
                
            end
        end
    end
    
end


% %% Plotting random points
% pts = 100;
% err = zeros(1,pts);
% x_random = 0.05*rand(1,pts);
% y_random = 0.25+0.1*rand(1,pts);
% z_random = 0.45+0.1*rand(1,pts);
% scatter3(x_random,y_random,z_random,'o','linewidth',1);
% hold on
% %scatter3(input_space(1,:),input_space(2,:),input_space(3,:),'c.','linewidth',1)
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% xlim([-0.1 0.1]);
% ylim([0.2 0.4]);
% zlim([0.4 0.6]);
% 
% h_random = plot3(V1(1),V1(2),V1(3),'m*');
% 
% random_traj = V1;
% input_space_random = [x_random;y_random;z_random];
% for i = 1:pts
%     i
%     % Finding out winning Neuron...........................................
%     dist = zeros(som_x,som_y,som_z);     %distances
%     for x = 1:som_x
%         for y = 1:som_y
%             for z = 1:som_z
%                 w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
%                 dist(x,y,z) = norm(w-input_space_random(:,i));
%             end
%         end
%     end
%     [C,I] = min(dist(:));
%     [I1,I2,I3] = ind2sub(size(dist),I);
%     index_min = [I1;I2;I3]; %index of winning neuron
%     
%     
%     % Finding output corresponding to current sample i.e current input.....
%     Theta_0_out = zeros(3,1);
%     h = zeros(som_x,som_y,som_z);        %neighbourhood functions
%     h_total = 0;                         %sum of all neighbourhood functions 's'
%     for x = 1:som_x
%         for y = 1:som_y
%             for z = 1:som_z
%                 w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
%                 eucl_dist = norm([x;y;z]-index_min);
%                 h(x,y,z) = exp(-(eucl_dist^2/(2*sigma^2)));
%                 h_total = h_total + h(x,y,z);
%                 
%                 Theta_lambda = [Theta(x,y,z,1);Theta(x,y,z,2);Theta(x,y,z,3)];
%                 
%                 A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
%                             A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
%                             A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
%                 
%                 W_lambda = w;    
%                 % Theta_0_out for coarse action
%                 Theta_0_out = Theta_0_out + h(x,y,z)*(Theta_lambda + A_lambda*(input_space_random(:,i)-w));
%             end
%         end
%     end
%     Theta_0_out = Theta_0_out/h_total;
%     
%     %Moving the Manipulator................................................
%     %This is the initial output which causes the coarse movement of the
%     %endeffector to reach a position having coordinates as V0.
%     %Coarse Action.........................................................
%     V0 = fwd_kin(Theta_0_out);
%     
%     %Fine Action...........................................................
%     Theta_temp = 0;
%     for x = 1:som_x %hence calculating in this for loop only
%         for y = 1:som_y
%             for z = 1:som_z              
%                 eucl_dist = norm([x;y;z]-index_min);        
%               
%                          
%                 A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
%                             A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
%                             A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
%                    
%                 % Theta_0_out for coarse action
%                 Theta_temp = Theta_temp + h(x,y,z)*(A_lambda*(input_space_random(:,i)-V0));               
%               
%             end
%         end
%     end
%     Theta_1_out = Theta_0_out + Theta_temp/h_total;  
%     V1 = fwd_kin(Theta_1_out);
% 
% 
%     random_traj = [random_traj V1];
%     err(i) = norm(input_space_random(:,i)-V1);
%     plotUpdate(h_random,random_traj(1,:),random_traj(2,:),random_traj(3,:));
%     pause(0.01);
% end
% 
% figure()
% plot(err)


%% Complicated Trajectory
% Note !! if you don't want to train start running from this section
% onwards.
load('VMC_data_27_Apr_9_36_AM_10000_samples.mat');
figure()
pts = 100;
x_st_line = linspace(-0.05,0.05,pts);
y_st_line = linspace(0.25,0.35,pts);
z_st_line = linspace(0.45,0.55,pts);
plot3(x_st_line,y_st_line,z_st_line,'b','linewidth',1);
hold on

dia = 0.1;
steps  = linspace(0,2*pi,pts);
x_cir = 0.05*cos(steps);
y_cir = 0.3+0.05*sin(steps);
%z_cir = 0.5*ones(1,100);
steps = linspace(0,12*pi,pts);
z_cir = 0.5+0.02*sin(steps);
plot3(x_cir,y_cir,z_cir,'b','linewidth',1);

steps  = linspace(0,2*pi,pts);
x_cir = 0.05*cos(steps);
y_cir = 0.3+0.05*sin(steps);
z_cir = 0.5*ones(1,100);
% steps = linspace(0,12*pi,pts);
% z_cir = 0.5+0.02*sin(steps);
plot3(x_cir,y_cir,z_cir,'b','linewidth',1);

%% Straight line Tracing................................................
%..........................................................................
pts = 100;
x_st_line = linspace(-0.05,0.05,pts);
y_st_line = linspace(0.25,0.35,pts);
z_st_line = linspace(0.45,0.55,pts);
plot3(x_st_line,y_st_line,z_st_line,'b','linewidth',1);
hold on
%scatter3(input_space(1,:),input_space(2,:),input_space(3,:),'c.','linewidth',1)
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([-0.1 0.1]);
ylim([0.2 0.4]);
zlim([0.4 0.6]);

h_st_line = plot3(V1(1),V1(2),V1(3),'mo');

st_line_traj = V1;
input_space_cir = [x_st_line;y_st_line;z_st_line];
for i = 1:pts
    i
    % Finding out winning Neuron...........................................
    dist = zeros(som_x,som_y,som_z);     %distances
    for x = 1:som_x
        for y = 1:som_y
            for z = 1:som_z
                w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
                dist(x,y,z) = norm(w-input_space_cir(:,i));
            end
        end
    end
    [C,I] = min(dist(:));
    [I1,I2,I3] = ind2sub(size(dist),I);
    index_min = [I1;I2;I3]; %index of winning neuron
    
    
    % Finding output corresponding to current sample i.e current input.....
    Theta_0_out = zeros(3,1);
    h = zeros(som_x,som_y,som_z);        %neighbourhood functions
    h_total = 0;                         %sum of all neighbourhood functions 's'
    for x = 1:som_x
        for y = 1:som_y
            for z = 1:som_z
                w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
                eucl_dist = norm([x;y;z]-index_min);
                h(x,y,z) = exp(-(eucl_dist^2/(2*sigma^2)));
                h_total = h_total + h(x,y,z);
                
                Theta_lambda = [Theta(x,y,z,1);Theta(x,y,z,2);Theta(x,y,z,3)];
                
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                
                W_lambda = w;    
                % Theta_0_out for coarse action
                Theta_0_out = Theta_0_out + h(x,y,z)*(Theta_lambda + A_lambda*(input_space_cir(:,i)-w));
            end
        end
    end
    Theta_0_out = Theta_0_out/h_total;
    
    %Moving the Manipulator................................................
    %This is the initial output which causes the coarse movement of the
    %endeffector to reach a position having coordinates as V0.
    %Coarse Action.........................................................
    V0 = fwd_kin(Theta_0_out);
    
    %Fine Action 1...........................................................
    Theta_temp = 0;
    for x = 1:som_x %hence calculating in this for loop only
        for y = 1:som_y
            for z = 1:som_z              
                eucl_dist = norm([x;y;z]-index_min);        
              
                         
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                   
                % Theta_0_out for coarse action
                Theta_temp = Theta_temp + h(x,y,z)*(A_lambda*(input_space_cir(:,i)-V0));               
              
            end
        end
    end
    Theta_1_out = Theta_0_out + Theta_temp/h_total;
    V1 = fwd_kin(Theta_1_out);
    
    %Fine Action 2...........................................................
    Theta_temp = 0;
    for x = 1:som_x %hence calculating in this for loop only
        for y = 1:som_y
            for z = 1:som_z              
                eucl_dist = norm([x;y;z]-index_min);        
              
                         
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                   
                % Theta_0_out for coarse action
                Theta_temp = Theta_temp + h(x,y,z)*(A_lambda*(input_space_cir(:,i)-V1));               
              
            end
        end
    end
    Theta_2_out = Theta_1_out + Theta_temp/h_total;
    V2 = fwd_kin(Theta_2_out);
    
    %Fine Action 3...........................................................
    Theta_temp = 0;
    for x = 1:som_x %hence calculating in this for loop only
        for y = 1:som_y
            for z = 1:som_z              
                eucl_dist = norm([x;y;z]-index_min);        
              
                         
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                   
                % Theta_0_out for coarse action
                Theta_temp = Theta_temp + h(x,y,z)*(A_lambda*(input_space_cir(:,i)-V2));               
              
            end
        end
    end
    Theta_3_out = Theta_2_out + Theta_temp/h_total;
   
    V3 = fwd_kin(Theta_3_out);
    st_line_traj = [st_line_traj V3];
    plotUpdate(h_st_line,st_line_traj(1,:),st_line_traj(2,:),st_line_traj(3,:));
    pause(0.01);
end




%% Circle Tracing.......................................................
%..........................................................................
pts = 100;
dia = 0.1;
steps  = linspace(0,2*pi,pts);
x_cir = 0.05*cos(steps);
y_cir = 0.3+0.05*sin(steps);
z_cir = 0.5*ones(1,100);
% steps = linspace(0,12*pi,pts);
% z_cir = 0.5+0.02*sin(steps);
plot3(x_cir,y_cir,z_cir,'b','linewidth',1);
hold on
%scatter3(input_space(1,:),input_space(2,:),input_space(3,:),'c.','linewidth',1)
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([-0.1 0.1]);
ylim([0.2 0.4]);
zlim([0.4 0.6]);

h_cir = plot3(V1(1),V1(2),V1(3),'mo');

cir_traj = V1;
input_space_cir = [x_cir;y_cir;z_cir];
for i = 1:pts
    i
    % Finding out winning Neuron...........................................
    dist = zeros(som_x,som_y,som_z);     %distances
    for x = 1:som_x
        for y = 1:som_y
            for z = 1:som_z
                w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
                dist(x,y,z) = norm(w-input_space_cir(:,i));
            end
        end
    end
    [C,I] = min(dist(:));
    [I1,I2,I3] = ind2sub(size(dist),I);
    index_min = [I1;I2;I3]; %index of winning neuron
    
    
    % Finding output corresponding to current sample i.e current input.....
    Theta_0_out = zeros(3,1);
    h = zeros(som_x,som_y,som_z);        %neighbourhood functions
    h_total = 0;                         %sum of all neighbourhood functions 's'
    for x = 1:som_x
        for y = 1:som_y
            for z = 1:som_z
                w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
                eucl_dist = norm([x;y;z]-index_min);
                h(x,y,z) = exp(-(eucl_dist^2/(2*sigma^2)));
                h_total = h_total + h(x,y,z);
                
                Theta_lambda = [Theta(x,y,z,1);Theta(x,y,z,2);Theta(x,y,z,3)];
                
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                
                W_lambda = w;    
                % Theta_0_out for coarse action
                Theta_0_out = Theta_0_out + h(x,y,z)*(Theta_lambda + A_lambda*(input_space_cir(:,i)-w));
            end
        end
    end
    Theta_0_out = Theta_0_out/h_total;
    
    %Moving the Manipulator................................................
    %This is the initial output which causes the coarse movement of the
    %endeffector to reach a position having coordinates as V0.
    %Coarse Action.........................................................
    V0 = fwd_kin(Theta_0_out);
    
    %Fine Action...........................................................
    Theta_temp = 0;
    for x = 1:som_x %hence calculating in this for loop only
        for y = 1:som_y
            for z = 1:som_z              
                eucl_dist = norm([x;y;z]-index_min);        
              
                         
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                   
                % Theta_0_out for coarse action
                Theta_temp = Theta_temp + h(x,y,z)*(A_lambda*(input_space_cir(:,i)-V0));               
              
            end
        end
    end
    Theta_1_out = Theta_0_out + Theta_temp/h_total;   
    V1 = fwd_kin(Theta_1_out);
    
    %Fine Action 2...........................................................
    Theta_temp = 0;
    for x = 1:som_x %hence calculating in this for loop only
        for y = 1:som_y
            for z = 1:som_z              
                eucl_dist = norm([x;y;z]-index_min);        
              
                         
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                   
                % Theta_0_out for coarse action
                Theta_temp = Theta_temp + h(x,y,z)*(A_lambda*(input_space_cir(:,i)-V1));               
              
            end
        end
    end
    Theta_2_out = Theta_1_out + Theta_temp/h_total;
    V2 = fwd_kin(Theta_2_out);
    
    cir_traj = [cir_traj V2];
    plotUpdate(h_cir,cir_traj(1,:),cir_traj(2,:),cir_traj(3,:));
    pause(0.01);
end


%% Sinusoid Tracking.......................................................
%..........................................................................
pts = 100;
dia = 0.1;
steps  = linspace(0,2*pi,pts);
x_cir = 0.05*cos(steps);
y_cir = 0.3+0.05*sin(steps);
%z_cir = 0.5*ones(1,100);
steps = linspace(0,12*pi,pts);
z_cir = 0.5+0.02*sin(steps);
plot3(x_cir,y_cir,z_cir,'b','linewidth',1);
hold on
%scatter3(input_space(1,:),input_space(2,:),input_space(3,:),'c.','linewidth',1)
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([-0.1 0.1]);
ylim([0.2 0.4]);
zlim([0.4 0.6]);

h_cir = plot3(V1(1),V1(2),V1(3),'mo');

cir_traj = V1;
input_space_cir = [x_cir;y_cir;z_cir];
for i = 1:pts
    i
    % Finding out winning Neuron...........................................
    dist = zeros(som_x,som_y,som_z);     %distances
    for x = 1:som_x
        for y = 1:som_y
            for z = 1:som_z
                w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
                dist(x,y,z) = norm(w-input_space_cir(:,i));
            end
        end
    end
    [C,I] = min(dist(:));
    [I1,I2,I3] = ind2sub(size(dist),I);
    index_min = [I1;I2;I3]; %index of winning neuron
    
    
    % Finding output corresponding to current sample i.e current input.....
    Theta_0_out = zeros(3,1);
    h = zeros(som_x,som_y,som_z);        %neighbourhood functions
    h_total = 0;                         %sum of all neighbourhood functions 's'
    for x = 1:som_x
        for y = 1:som_y
            for z = 1:som_z
                w = [weights(x,y,z,1);weights(x,y,z,2);weights(x,y,z,3)];
                eucl_dist = norm([x;y;z]-index_min);
                h(x,y,z) = exp(-(eucl_dist^2/(2*sigma^2)));
                h_total = h_total + h(x,y,z);
                
                Theta_lambda = [Theta(x,y,z,1);Theta(x,y,z,2);Theta(x,y,z,3)];
                
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                
                W_lambda = w;    
                % Theta_0_out for coarse action
                Theta_0_out = Theta_0_out + h(x,y,z)*(Theta_lambda + A_lambda*(input_space_cir(:,i)-w));
            end
        end
    end
    Theta_0_out = Theta_0_out/h_total;
    
    %Moving the Manipulator................................................
    %This is the initial output which causes the coarse movement of the
    %endeffector to reach a position having coordinates as V0.
    %Coarse Action.........................................................
    V0 = fwd_kin(Theta_0_out);
    
    %Fine Action...........................................................
    Theta_temp = 0;
    for x = 1:som_x %hence calculating in this for loop only
        for y = 1:som_y
            for z = 1:som_z              
                eucl_dist = norm([x;y;z]-index_min);        
              
                         
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                   
                % Theta_0_out for coarse action
                Theta_temp = Theta_temp + h(x,y,z)*(A_lambda*(input_space_cir(:,i)-V0));               
              
            end
        end
    end
    Theta_1_out = Theta_0_out + Theta_temp/h_total;   
    V1 = fwd_kin(Theta_1_out);
    
    %Fine Action 2...........................................................
    Theta_temp = 0;
    for x = 1:som_x %hence calculating in this for loop only
        for y = 1:som_y
            for z = 1:som_z              
                eucl_dist = norm([x;y;z]-index_min);        
              
                         
                A_lambda = [A(x,y,z,1,1) A(x,y,z,1,2) A(x,y,z,1,3);
                            A(x,y,z,2,1) A(x,y,z,2,2) A(x,y,z,2,3);
                            A(x,y,z,3,1) A(x,y,z,3,2) A(x,y,z,3,3)];
                   
                % Theta_0_out for coarse action
                Theta_temp = Theta_temp + h(x,y,z)*(A_lambda*(input_space_cir(:,i)-V1));               
              
            end
        end
    end
    Theta_2_out = Theta_1_out + Theta_temp/h_total;
    V2 = fwd_kin(Theta_2_out);
    
    cir_traj = [cir_traj V2];
    plotUpdate(h_cir,cir_traj(1,:),cir_traj(2,:),cir_traj(3,:));
    pause(0.01);
end






