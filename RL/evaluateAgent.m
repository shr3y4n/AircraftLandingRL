function [eval_results, flight_log] = evaluateAgent(agent, P, wind_cfg, custom_init)
% EVALUATEAGENT Evaluates a trained RL landing policy deterministically.
%
% Runs a full landing trajectory evaluation without exploration noise,
% records complete flight telemetry and logs performance metrics.
%
% Inputs:
%   agent       - Trained policy agent (rlEngine struct or RL Toolbox agent)
%   P           - Aircraft parameters struct (optional)
%   wind_cfg    - Wind disturbance configuration (optional)
%   custom_init - Custom initial condition struct (optional)
%
% Outputs:
%   eval_results - Struct of quantitative landing metrics (metrics.m)
%   flight_log   - Comprehensive time-series trajectory struct for plotting

if nargin < 2 || isempty(P),          P = aircraftParameters(); end
if nargin < 3 || isempty(wind_cfg),   wind_cfg.mode = 'none';   end
if nargin < 4,                        custom_init = [];         end

% Instantiate deterministic evaluation environment
env = AircraftLandingEnv(P, wind_cfg, false);
obs = env.reset(custom_init);

max_steps = env.max_steps;

% Preallocate flight logging arrays
time_hist    = zeros(max_steps, 1);
state_hist   = zeros(6, max_steps);
ctrl_hist    = zeros(2, max_steps);
ref_hist     = zeros(4, max_steps); % [h_ref; V_ref; hdot_ref; theta_ref]
wind_hist    = zeros(2, max_steps);
reward_hist  = zeros(max_steps, 1);

step_idx = 0;
done = false;
last_info = [];

while ~done && (step_idx < max_steps)
    step_idx = step_idx + 1;
    
    % Deterministic action selection (zero exploration noise)
    if isstruct(agent) && isfield(agent, 'actor')
        action = rlEngine('getAction', agent, obs, false);
    elseif exist('getAction', 'file') == 2
        action = getAction(agent, {obs});
        if iscell(action), action = action{1}; end
    else
        action = [0; 0];
    end
    
    % Record pre-step state and reference
    cur_state = env.state;
    ref = landingScenario('reference', cur_state(1), cur_state(2), P);
    
    [w_vel, ~] = windModel(env.current_time, cur_state, wind_cfg);
    
    time_hist(step_idx)       = env.current_time;
    state_hist(:, step_idx)   = cur_state;
    ref_hist(:, step_idx)     = [ref.h_ref; ref.V_ref; ref.hdot_ref; ref.theta_ref];
    wind_hist(:, step_idx)    = w_vel;
    
    % Step environment
    [next_obs, reward, done, info] = env.step(action);
    
    ctrl_hist(:, step_idx)   = info.controls;
    reward_hist(step_idx)    = reward;
    last_info                = info;
    
    obs = next_obs;
end

% Trim unallocated tail
time_hist   = time_hist(1:step_idx);
state_hist  = state_hist(:, 1:step_idx);
ctrl_hist   = ctrl_hist(:, 1:step_idx);
ref_hist    = ref_hist(:, 1:step_idx);
wind_hist   = wind_hist(:, 1:step_idx);
reward_hist = reward_hist(1:step_idx);

% Package Flight Telemetry Log
flight_log.time      = time_hist;
flight_log.x         = state_hist(1, :);
flight_log.h         = state_hist(2, :);
flight_log.V         = state_hist(3, :);
flight_log.gamma     = state_hist(4, :);
flight_log.theta     = state_hist(5, :);
flight_log.q         = state_hist(6, :);
flight_log.alpha     = flight_log.theta - flight_log.gamma;
flight_log.sink_rate = -flight_log.V .* sin(flight_log.gamma);

flight_log.delta_e   = ctrl_hist(1, :);
flight_log.delta_T   = ctrl_hist(2, :);

flight_log.h_ref     = ref_hist(1, :);
flight_log.V_ref     = ref_hist(2, :);
flight_log.hdot_ref  = ref_hist(3, :);
flight_log.theta_ref = ref_hist(4, :);

flight_log.wx        = wind_hist(1, :);
flight_log.wh        = wind_hist(2, :);
flight_log.reward    = reward_hist;
flight_log.status    = last_info.status;
flight_log.is_success= last_info.is_success;

% Compute Standardized Quantitative Metrics
eval_results = metrics(flight_log, P);

end
