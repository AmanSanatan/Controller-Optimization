% Initialization
clear ; close all; clc
%--------------------------------------------------------------------------
%Importing the training data
load("rpm.csv");
load("torque.csv");
load("voltage.csv");
rpm_train_present = rpm(3:floor(size(rpm)*0.6));
rpm_train_past = rpm(2:floor(size(rpm)*0.6)-1);
rpm_train_past1 = rpm(1:floor(size(rpm)*0.6)-2);
rpm_val = rpm(floor(size(rpm)*0.6)+1:floor(size(rpm)*0.8));
rpm_test = rpm(floor(size(rpm)*0.8)+1:floor(size(rpm)));
voltage_train_past = voltage(1:floor(size(voltage)*0.6-2));
voltage_train_present =voltage(2:floor(size(voltage)*0.6-1));
voltage_val = voltage(floor(size(voltage)*0.6)+1:floor(size(voltage)*0.8));
voltage_test = voltage(floor(size(voltage)*0.8)+1:floor(size(voltage)));
torque_train_past = torque(1:floor(size(torque)*0.6-2));
torque_train_present =torque(3:floor(size(torque)*0.6));
torque_val = torque(floor(size(torque)*0.6)+1:floor(size(torque)*0.8));
torque_test = torque(floor(size(torque)*0.8)+1:floor(size(torque)));
%--------------------------------------------------------------------------
%Fixed parameters
input_layer_size = 1;
output_layer_size = 1;
hidden_layer_count = 1;
alpha = 0.1;
cost = [];
cost1 = [];
x =1;
average_input = mean(rpm_train_present,1);
%--------------------------------------------------------------------------
%State space model of dc motor
R = 1;                % Ohms
L = 0.5;                % Henrys
Km = 0.1;               % torque constant
Kb = 0.1;               % back emf constant
Kf = 0.2;               % Nms
J = 0.02;               % kg.m^2/s^2

h1 = tf(Km,[L R]);            % armature
h2 = tf(1,[J Kf]);            % eqn of motion
dcm = ss(h2) * [h1 , 1];      % w = h2 * (h1*Va + Td)
dcm = feedback(dcm,Kb,1,1);   % close back emf loop
Kff = 1/dcgain(dcm(1));
dcm = dcm * diag([Kff,1]); 
%--------------------------------------------------------------------------
%Generating rpm
rpm = lsim(dcm,[(voltage(2:end))';torque(2:end)'],0:0.2:0.2*size(voltage,1)-0.4);

rpm_train_present = rpm(2:floor(size(rpm)*0.6)-1);
rpm_train_past = rpm(1:floor(size(rpm)*0.6-2));
rpm_train_past1 = rpm(1:floor(size(rpm)*0.6)-2);
rpm_val = rpm(floor(size(rpm)*0.6)+1:floor(size(rpm)*0.8));
rpm_test = rpm(floor(size(rpm)*0.8)+1:floor(size(rpm)));
plot(rpm)
hold
plot(voltage)
%--------------------------------------------------------------------------

%Input variables
X0 = ([(rpm_train_past - mean(rpm_train_past))/std(rpm_train_past),(voltage_train_present-mean(voltage_train_present))/std(voltage_train_present)]);
Y = ((rpm_train_present - mean(rpm_train_present))/std(rpm_train_present));
%--------------------------------------------------------------------------
%Training the neural networks
[W0, W1, cost] = oneLayerNetwork( alpha, X0, Y);
[ WC0, WC1 ] = controller_network( X0, Y, W0, W1 );
%--------------------------------------------------------------------------
%Getting the test set predictions
rpm_in = load("step.csv");;

[ Y_out, voltage_out ] = controller_with_NNmodelPredictor( rpm_in, WC0, WC1, W0, W1);
%--------------------------------------------------------------------------
% Simulation with trained controller and model

% Reference RPM input (single value)
ref_rpm = 1000; % example reference RPM value

% Normalize the reference RPM using training data statistics
ref_rpm_norm = (ref_rpm - mean(rpm_train_present))/std(rpm_train_present);

% Initial conditions
rpm_actual = 0; % starting from 0 RPM
rpm_actual_norm = (rpm_actual - mean(rpm_train_present))/std(rpm_train_present);
voltage_out = 0; % initial voltage

% Simulation parameters
sim_time = 100; % number of time steps to simulate
time_history = zeros(sim_time, 1);
rpm_history = zeros(sim_time, 1);
voltage_history = zeros(sim_time, 1);

for t = 1:sim_time
    % Current state (normalized)
    X = [(rpm_actual_norm - mean(rpm_train_past))/std(rpm_train_past), ...
         (voltage_out - mean(voltage_train_present))/std(voltage_train_present)];
    
    % Controller generates voltage command
    [voltage_norm, ~] = controller_network_predict(X, ref_rpm_norm, WC0, WC1);
    voltage_out = voltage_norm * std(voltage_train_present) + mean(voltage_train_present);
    
    % Plant model prediction (using your neural network model)
    X_model = [rpm_actual_norm, voltage_norm];
    [rpm_pred_norm, ~] = oneLayerNetworkPredict(X_model, W0, W1);
    rpm_actual_norm = rpm_pred_norm;
    
    % Denormalize for recording
    rpm_actual = rpm_pred_norm * std(rpm_train_present) + mean(rpm_train_present);
    
    % Store results
    time_history(t) = t;
    rpm_history(t) = rpm_actual;
    voltage_history(t) = voltage_out;
end

% Plot results
figure;
subplot(2,1,1);
plot(time_history, rpm_history, 'b', 'LineWidth', 2);
hold on;
plot([0 sim_time], [ref_rpm ref_rpm], 'r--', 'LineWidth', 1.5);
ylabel('RPM');
title('System Response');
legend('Actual RPM', 'Reference RPM');

subplot(2,1,2);
plot(time_history, voltage_history, 'g', 'LineWidth', 2);
ylabel('Voltage');
xlabel('Time Step');