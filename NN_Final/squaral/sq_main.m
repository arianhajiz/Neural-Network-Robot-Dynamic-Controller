
%% Clearing Existing Data
clc
clear all
close all

%% Mobile Robot Parameters
R = 0.05; % Wheel radius
L = 0.1; % Distance between wheels
r = 0.05; % Radius of each wheel
m = 15;   % Robot mass
I = 5;    % Moment of inertia

%% Simulation parameters 
sample_time = 1;       % Reducing the sample time can cause higher CPU usage
SimulationTime = 100;     % Change this if you want to change the simulation duration
counter = 0;
time = 0:sample_time:SimulationTime;

%% Initial state
q = [0, 0, 0];  % Initial state [x_c, y_c, theta]
q_dot = [0, 0, 0];          % Initial linear velocity
omega = 0;      % Initial angular velocity

%% Parameters of the Circular Trajectory
a = 2;       % Circle Center in meters (X Axis)
c = 2;       % Circle Center in meters (Y Axis)
b = 2;       % Radius in meters
omega_r = 0.1; % Angular velocity for theta reference

%% Desired Trajectory (You can Change The Desired Trajectory)
% Define parameters for the square trajectory
side_length = 5;   % side length of the square
num_points = 200;  % number of points in the trajectory
points_per_side = num_points / 4;  % points per side

% Generate the x and y coordinates for each side of the square
x = [];
y = [];
theta = [];

% Bottom side (moving right)
x = [x, linspace(0, side_length, points_per_side)];
y = [y, zeros(1, points_per_side)];

% Right side (moving up)
x = [x, side_length * ones(1, points_per_side)];
y = [y, linspace(0, side_length, points_per_side)];

% Top side (moving left)
x = [x, linspace(side_length, 0, points_per_side)];
y = [y, side_length * ones(1, points_per_side)];


% Left side (moving down)
x = [x, zeros(1, points_per_side)];
y = [y, linspace(side_length, 0, points_per_side)];
for i = 1:size(x,2)
    if(x(i) == 0 && y(i) > 0)
        theta(1,i) = 0;
    elseif(x(i) == 0 && y(i) < 0)
        theta(1,i) = -pi;
    else 
        theta(1,i) = atan(y/x);
    end
end

q_ref = [x', y', theta'];

q_dot_ref(1,:) = [0 0 0];
for i = 2:size(q_ref,1)
    q_dot_ref(i,1) = (q_ref(i,1) - q_ref(i-1,1))/sample_time;
    q_dot_ref(i,2) = (q_ref(i,2) - q_ref(i-1,2))/sample_time;
    q_dot_ref(i,3) = (q_ref(i,3) - q_ref(i-1,3))/sample_time;
end

q_ddot_ref(1,:) = [0 0 0];
for i = 2:size(q_ref,1)
    q_ddot_ref(i,1) = (q_dot_ref(i,1) - q_dot_ref(i-1,1))/sample_time;
    q_ddot_ref(i,2) = (q_dot_ref(i,2) - q_dot_ref(i-1,2))/sample_time;
    q_ddot_ref(i,3) = (q_dot_ref(i,3) - q_dot_ref(i-1,3))/sample_time;
end

plot(q_ref(:,1),q_ref(:,2))
%Tobe or not To be
q = q_ref(1,:);
q_dot = q_dot_ref(1,:);

%% Improved Training Data Generation
K = 1;
iterations = 100;
lambda = 0.08;
max_K = 20;
for i = 1:size(q_ref,1)
    e = q_ref(i,:) - q;

    q_all(i,:) = q;
    q_dot_all(i,:) = q_dot;
    e_all(i,:) = e;
    X_train{i} = [q_ref(1:i,:),q_dot_ref(1:i,:),e_all(1:i,:),q_dot_all(1:i,:)];

    if(K > max_K)
        K = max_K;
    end
    centers = myKmeans(X_train{i}, K);
    
    max_d = 0;
    for temp = 1:K
        for temp2 = 1:K
            if(norm(centers(temp2)-centers(temp)) > max_d)
                max_d = norm(centers(temp2)-centers(temp));
            end
        end
    end
    % getting sigmas according to handouts and the book with K-means and RLS
    sigma = max_d*max(max(q_ref))/sqrt(2*K)  ;
    
    % this function is defined at the end of the file
    [out, Phi] = rbf_HL(X_train{i}, centers, sigma);
    
    % adding bias to the weights 
    Phi = [Phi , ones(size(X_train{i},1),1)];

    
    
    P = lambda * eye(K + 1);
    w = zeros(K + 1, 2); % Initialize weight matrix for each output dimension

    for n = 1:i
        % Update P matrix
        P = P - (P * Phi(n,:)' * Phi(n,:) * P) / (1 + Phi(n,:) * P * Phi(n,:)');
        
        % Calculate gain vector
        g = P * Phi(n,:)';
        
        % Prediction error
        [M, V, G, B, A] = robot_dynamics(q_ref(i,:), q_dot_ref(i,:));
        tau_ref = pinv(B)*(M*q_ddot_ref(i,:)' + V + G - A);
        pre = tau_ref' - Phi(n,:) * w;
        
        % Update weights
        w = w + g * pre;
    end
    
    % Calculate mean squared error
    T = Phi * w;
    if(i>1)
        [t,x] = ode45(@(t,x) odefcn(t,x,T), sample_time*[i-1 i], [q_ref(i-1,:)';q_dot_ref(i-1,:)']);
        q = ([x(length(t),1),x(length(t),2),x(length(t),3)]);
        q_dot = [x(length(t),4),x(length(t),5),x(length(t),6)];
    end
    
end

%% Plotting Output Data
figure;
hold on 
plot(q_ref(1:size(q_all,1),1),q_ref(1:size(q_all,1),2),'b',LineWidth=2)
plot(q_all(:,1),q_all(:,2),"r--",LineWidth=2);
legend(["reference" "predicted"]);
hold off

%% function definitions
function [output, Phi] = rbf_HL(X, centers, sigma)
    num_data = size(X, 1);
    num_centers = size(centers, 1);
    Phi = zeros(num_data, num_centers);
    
    for i = 1:num_data
        for j = 1:num_centers
            Phi(i, j) = exp(-norm(X(i,:) - centers(j,:))^2 / (2 * sigma^2));
        end
    end
    
    output = Phi; % Just return Phi if we don't have target values
end

function [centers] = myKmeans(X, k)
    % Randomly initialize the cluster centers
    num_samples = size(X, 1);
    random_indices = randperm(num_samples, k);
    centers = X(random_indices, :);

    % Initialize variables
    cluster_assignment = zeros(num_samples, 1);
    max_iters = 100;
    iter = 0;

    while iter < max_iters
        iter = iter + 1;

        % Assign each sample to the nearest center
        for i = 1:num_samples
            distances = sum((X(i, :) - centers) .^ 2, 2);
            [~, min_index] = min(distances);
            cluster_assignment(i) = min_index;
        end

        % Update centers
        new_centers = zeros(size(centers));
        for j = 1:k
            cluster_points = X(cluster_assignment == j, :);
            if ~isempty(cluster_points)
                new_centers(j, :) = mean(cluster_points, 1);
            else
                % Reinitialize empty cluster
                new_centers(j, :) = X(randi(num_samples), :);
            end
        end

        % Check for convergence
        if all(new_centers == centers)
            break;
        end

        centers = new_centers;
    end
end

function [M, V, G, B, A] = robot_dynamics(q, q_dot)
    % Robot parameters
    m = 15; % kg
    I = 5; % kg*m^2
    R = 0.15; % m
    r = 0.05; % m
    d = 0.1; % m
    
    % State variables
    theta = q(3);
    theta_dot = q_dot(3);
    
    % Inertia matrix
    M = [m 0 m*d*sin(theta);...
        0 m -m*d*cos(theta);...
        m*d*sin(theta) -m*d*cos(theta) I];
    
    % Coriolis and centrifugal matrix
    V = [m*d*cos(theta)*theta_dot^2;... 
         m*d*sin(theta)*theta_dot^2;...
         0];
    
    % Gravity vector
    G = [0; 0; 0]; % Assuming no gravity effects in x and y directions
    
    % Input matrix
    B = (1/r)*[cos(theta) cos(theta);...
        sin(theta) sin(theta) ;...
        R -R];

    A = [-m*sin(theta)*(q_dot(1)*cos(theta) + q_dot(2)*sin(theta))*theta_dot;...
        m*cos(theta)*(q_dot(1)*cos(theta) + q_dot(2)*sin(theta))*theta_dot;...
        -d*m*(q_dot(1)*cos(theta) + q_dot(2)*sin(theta))*theta_dot];

    S = [sin(theta) -d*cos(theta);cos(theta) -d*sin(theta);0 1];
end

function state = odefcn(t,x,T)
   
    state = zeros(6,1);

    % Robot parameters
    m = 15; % kg
    I = 5; % kg*m^2
    R = 0.15; % m
    r = 0.05; % m
    d = 0.1; % m
    
    % State variables
    theta = x(3);
    theta_dot = x(6);
    
    % Inertia matrix
    M = [m 0 m*d*sin(theta);...
        0 m -m*d*cos(theta);...
        m*d*sin(theta) -m*d*cos(theta) I];
    
    % Coriolis and centrifugal matrix
    V = [m*d*cos(theta)*theta_dot^2;... 
         m*d*sin(theta)*theta_dot^2;...
         0];
    
    % Gravity vector
    G = [0; 0; 0]; % Assuming no gravity effects in x and y directions
    
    % Input matrix
    B = (1/r)*[cos(theta) cos(theta);...
        sin(theta) sin(theta) ;...
        R -R];

    A = [-m*sin(theta)*(x(4)*cos(theta) + x(5)*sin(theta))*theta_dot;...
        m*cos(theta)*(x(4)*cos(theta) + x(5)*sin(theta))*theta_dot;...
        -d*m*(x(4)*cos(theta) + x(5)*sin(theta))*theta_dot];
    

    eqn = pinv(M)*(B*T(end,:)' + A - G - V);

    state(1) = x(4);
    state(2) = x(5);
    state(3) = x(6);

    state(4) = eqn(1,1);
    state(5) = eqn(2,1);
    state(6) = eqn(3,1);
      
end