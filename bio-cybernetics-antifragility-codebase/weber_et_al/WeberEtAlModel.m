% Sigma-Pi Neuronal model for sensorimotor learning of Weber et al.

%% SETUP ENV
clear all; close all; clc
%% INPUT DATA
MIN_VAL_IN = 0.0;
MAX_VAL_IN = 1.0;
MIN_VAL_OUT = 0.0;
MAX_VAL_OUT = 2.0;

% number of neurons in each population
N_NEURONS = 15;
% maximum initialization range of weights and activities in the populations
MIN_INIT_RANGE = -0.5;
MAX_INIT_RANGE = 0.5;
% learning rate
ETA = 0.001;
% number of training epochs
MAX_EPOCHS = 199999;

% the neighborhood kernel shrinks in time

% simulation time vector
t = 1:MAX_EPOCHS;
%initial neighborhood function width
SIGMA0  = N_NEURONS/2;
% exponential decay
SIGMAT = SIGMA0*exp(-t./(MAX_EPOCHS/log(SIGMA0)));

% or the dedicated setup suggested in the paper
% t1 = 1:33000;
% SIGMAT(t1) = -0.000136364*t1 + 8;
% t2 = 33001:199999;
% SIGMAT(t2) = -0.000020958*t2 + 4.191616766;
% t3 = 200001:MAX_EPOCHS;
% SIGMAT(t3) = 0.0;

% normalalization factor for activities
NORM_ACT = 1;

% network structure (populations)
population_x = struct('lsize', N_NEURONS, ...
    'A', rand(N_NEURONS, 1));
population_y = struct('lsize', N_NEURONS, ...
    'A', rand(N_NEURONS, 1));
population_x_bar = struct('lsize', N_NEURONS, ...
    'A', rand(N_NEURONS, 1));
population_y_bar = struct('lsize', N_NEURONS, ...
    'A', rand(N_NEURONS, 1));
population_z = struct('lsize', N_NEURONS,...
    'A', rand(N_NEURONS, 1), ...
    'W', MIN_INIT_RANGE + (MAX_INIT_RANGE - MIN_INIT_RANGE)*rand(N_NEURONS, N_NEURONS, N_NEURONS));

% handler for changes in activity in output layer
delta_act = zeros(N_NEURONS, N_NEURONS);

% allocate the input data vectors
val_z = zeros(MAX_EPOCHS, 1);
val_y = zeros(MAX_EPOCHS, 1);
val_x = zeros(MAX_EPOCHS, 1);
val_x_bar = zeros(MAX_EPOCHS, 1);
val_y_bar = zeros(MAX_EPOCHS, 1);

% preset systematic data as described in paper
for it =1:MAX_EPOCHS
    val_y(it) = MIN_VAL_IN + (MAX_VAL_IN - MIN_VAL_IN)*rand;
end
val_y = sort(val_y);

%% NETWORK SIMULATION
for t = 1:MAX_EPOCHS
    
    % we have systematically initialized data as 
    % to satisfy val_x.^3 + val_y = val_z
    val_x(t) = 2*MAX_VAL_IN/3 + (MAX_VAL_IN - 2*MAX_VAL_IN/3)*rand;
    val_z(t) = val_x(t).^3 + val_y(t);
    
    % produce the population code for (x, y) pair, post-synaptic
    population_x.A = population_encoder(val_x(t), MAX_VAL_IN, N_NEURONS);
    population_y.A = population_encoder(val_y(t), MAX_VAL_IN, N_NEURONS);
    
    % forward propagation of activities in populations x and y to z
    for idx = 1:N_NEURONS % in z population
        for jdx = 1:N_NEURONS % in x population
            for kdx = 1:N_NEURONS % in y population
                delta_act(jdx, kdx) = population_z.W(idx, jdx, kdx)*population_x.A(jdx)*population_y.A(kdx);
            end
        end
        population_z.A(idx) = sum(delta_act(:));
    end
    
    % find the best matching unit
    % but first init with 0
    bmu_pos = 0;
    [bmu_val, bmu_pos] = max(conv(population_z.A, gauss_kernel(NORM_ACT, N_NEURONS, SIGMAT(t), bmu_pos), 'same'));
    
    % compute activations in the output population z
    population_z.A = gauss_kernel(NORM_ACT, N_NEURONS, SIGMAT(t), bmu_pos);
    
    % we have systematically initialized data as described in paper such
    % that randomly selected (val_x_bar, val_y_bar) satisfy val_x_bar + val_y_bar = val_z
    val_x_bar(t) = MIN_VAL_IN + (val_z(t))*rand;
    val_y_bar(t) = val_z(t) - val_x_bar(t).^3;
    
    % produce the population code for (x_bar, y_bar) pair, pre-synaptic
    population_x_bar.A = population_encoder(val_x_bar(t), MAX_VAL_IN, N_NEURONS);
    population_y_bar.A = population_encoder(val_y_bar(t), MAX_VAL_IN, N_NEURONS);
    
    % apply Hebbian learning to update weights
    for idx = 1:N_NEURONS % in z population
        for jdx = 1:N_NEURONS % in x population
            for kdx = 1:N_NEURONS % in y population
                population_z.W(idx, jdx, kdx) = population_z.W(idx, jdx, kdx) + ...
                    ETA*population_z.A(idx)*(population_x_bar.A(jdx)*population_y_bar.A(kdx) - population_z.W(idx, jdx, kdx));
            end
        end
    end
    
    if any(isnan(population_z.W(:)))
                    uiwait(warndlg('Check your data'));
    end
                
end % end of input dataset

%% VISUALIZATION
figure; set(gcf, 'color', 'w');
for idx = 1:N_NEURONS
    subplot(1, N_NEURONS, idx);
    Wm(:,:) = population_z.W(idx, :, :);
    pcolor(Wm); colorbar;
end