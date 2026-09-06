function [flight_log, landing_metrics] = simulatePID(P, wind_cfg, custom_init, max_time)
% SIMULATEPID Runs closed-loop aircraft landing simulation under classical PID control.
%
% Executes the multi-loop longitudinal flight control system (pitch attitude,
% glide-slope outer loop, flare law, and autothrottle) through approach,
% flare, and touchdown.
%
% Inputs:
%   P           - Aircraft parameters struct (optional)
%   wind_cfg    - Wind disturbance configuration struct (optional)
%   custom_init - Custom initial state options struct (optional)
%   max_time    - Maximum simulation time [s] (defaults to P.t_max)
%
% Outputs:
%   flight_log      - Complete time-series flight telemetry struct
%   landing_metrics - Quantitative performance metrics struct from metrics.m

if nargin < 1 || isempty(P),          P = aircraftParameters(); end
if nargin < 2 || isempty(wind_cfg),   wind_cfg.mode = 'none';   end
if nargin < 3,                        custom_init = [];         end
if nargin < 4 || isempty(max_time),   max_time = P.t_max;       end

%% 1. Initialize Aircraft State and Controller
[init_state, trim] = initializeAircraft(P, custom_init);
ctrl_state = pidLandingController('reset', P, trim);
sensor_state = [];

cur_state = [init_state.x; init_state.h; init_state.V; init_state.gamma; ...
             init_state.theta; init_state.q];

dt = P.dt;
max_steps = round(max_time / dt);

% Preallocate time-series logging arrays
time_hist    = zeros(max_steps, 1);
state_hist   = zeros(6, max_steps);
ctrl_hist    = zeros(2, max_steps);
ref_hist     = zeros(4, max_steps); % [h_ref; V_ref; hdot_ref; theta_ref]
wind_hist    = zeros(2, max_steps);
aero_hist    = zeros(4, max_steps); % [CL; CD; Cm; alpha]

step = 0;
done = false;
last_status = 'IN_FLIGHT';
is_success  = false;

%% 2. Closed-Loop Simulation Step
for k = 1:max_steps
    step = step + 1;
    t = (k - 1) * dt;
    
    % Query Reference Trajectory
    ref = landingScenario('reference', cur_state(1), cur_state(2), P);
    
    % Query Atmospheric Wind
    [w_vel, ~] = windModel(t, cur_state, wind_cfg);
    
    % Sensor Measurement Model
    [meas, sensor_state] = sensorModel(cur_state, P, sensor_state);
    
    % Compute Control Commands via Classical PID Flight Controller
    [controls, ctrl_state, ~] = pidLandingController('step', meas, ref, P, trim, ctrl_state);
    
    % Record Telemetry
    time_hist(step)     = t;
    state_hist(:, step) = cur_state;
    ctrl_hist(:, step)  = controls;
    ref_hist(:, step)   = [ref.h_ref; ref.V_ref; ref.hdot_ref; ref.theta_ref];
    wind_hist(:, step)  = w_vel;
    
    % Advance Physical System Dynamics via RK4
    [next_state, dyn_info] = updateAircraft(cur_state, controls, dt, P, wind_cfg);
    
    aero_hist(:, step) = [dyn_info.CL; dyn_info.CD; dyn_info.Cm; dyn_info.alpha];
    
    % Check Landing and Flight Safety Boundaries
    landing_status = landingScenario('evaluate', next_state, P);
    
    cur_state   = next_state;
    last_status = landing_status.status;
    is_success  = landing_status.details.is_success;
    
    if landing_status.is_terminal
        done = true;
        break;
    end
end

%% 3. Package Flight Telemetry Log
time_hist   = time_hist(1:step);
state_hist  = state_hist(:, 1:step);
ctrl_hist   = ctrl_hist(:, 1:step);
ref_hist    = ref_hist(:, 1:step);
wind_hist   = wind_hist(:, 1:step);
aero_hist   = aero_hist(:, 1:step);

flight_log.time       = time_hist;
flight_log.x          = state_hist(1, :);
flight_log.h          = state_hist(2, :);
flight_log.V          = state_hist(3, :);
flight_log.gamma      = state_hist(4, :);
flight_log.theta      = state_hist(5, :);
flight_log.q          = state_hist(6, :);
flight_log.alpha      = aero_hist(4, :);
flight_log.sink_rate  = -flight_log.V .* sin(flight_log.gamma);

flight_log.delta_e    = ctrl_hist(1, :);
flight_log.delta_T    = ctrl_hist(2, :);

flight_log.h_ref      = ref_hist(1, :);
flight_log.V_ref      = ref_hist(2, :);
flight_log.hdot_ref   = ref_hist(3, :);
flight_log.theta_ref  = ref_hist(4, :);

flight_log.wx         = wind_hist(1, :);
flight_log.wh         = wind_hist(2, :);
flight_log.CL         = aero_hist(1, :);
flight_log.CD         = aero_hist(2, :);
flight_log.Cm         = aero_hist(3, :);

flight_log.status     = last_status;
flight_log.is_success = is_success;

%% 4. Compute Quantitative Landing Metrics
landing_metrics = metrics(flight_log, P);

end
