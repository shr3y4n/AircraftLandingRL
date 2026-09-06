function [r, breakdown] = rewardFunction(state, controls, prev_controls, P, landing_status)
% REWARDFUNCTION Computes research-grade reward signal for landing RL agent.
%
% Shapes continuous tracking rewards (glide-slope, airspeed, sink rate, pitch)
% and applies calibrated terminal penalties and bonuses for touchdown quality.
% Avoids reward exploitation (e.g. loitering or climbing forever) by incorporating
% forward progress incentives and corridor boundaries.
%
% Inputs:
%   state          - Current aircraft state (struct or [6x1] vector)
%   controls       - Current control action [delta_e, delta_T]
%   prev_controls  - Previous control action [delta_e_prev, delta_T_prev]
%   P              - Parameter struct from aircraftParameters()
%   landing_status - Struct from landingScenario('evaluate', state, P)
%
% Outputs:
%   r         - Total scalar reward for the current step
%   breakdown - Struct detailing individual reward components for analysis

if nargin < 4 || isempty(P)
    P = aircraftParameters();
end

if isstruct(state)
    x     = state.x;
    h     = state.h;
    V     = state.V;
    gamma = state.gamma;
    theta = state.theta;
    q     = state.q;
else
    x     = state(1);
    h     = state(2);
    V     = state(3);
    gamma = state(4);
    theta = state(5);
    q     = state(6);
end

if isstruct(controls)
    de = controls.elevator;
    dT = controls.throttle;
else
    de = controls(1);
    dT = controls(2);
end

if nargin < 3 || isempty(prev_controls)
    de_prev = de;
    dT_prev = dT;
elseif isstruct(prev_controls)
    de_prev = prev_controls.elevator;
    dT_prev = prev_controls.throttle;
else
    de_prev = prev_controls(1);
    dT_prev = prev_controls(2);
end

if nargin < 5 || isempty(landing_status)
    landing_status = landingScenario('evaluate', [x; h; V; gamma; theta; q], P);
end

%% 1. Continuous Tracking Errors
ref = landing_status.details.ref;
e_h = ref.h_ref - h;
e_V = ref.V_ref - V;

% Normalized tracking quadratic costs
norm_eh = e_h / P.rl.obs_norm.eh;
norm_eV = e_V / P.rl.obs_norm.eV;

r_glideslope = -P.reward.w_eh * (norm_eh^2);
r_airspeed   = -P.reward.w_eV * (norm_eV^2);

%% 2. Flight Attitude and Stability Costs
alpha = theta - gamma;
sink_rate = -V * sin(gamma);

% Penalize sink rate deviation from reference
e_hdot = ref.hdot_ref - (-sink_rate);
norm_hdot = e_hdot / P.rl.obs_norm.hdot;
r_sink = -P.reward.w_hdot * (norm_hdot^2);

% Penalize excessive pitch deviation from reference
e_theta = theta - ref.theta_ref;
r_pitch = -P.reward.w_theta * (e_theta^2);

% Heavy penalty when angle of attack nears or exceeds stall
r_alpha = 0.0;
if alpha > deg2rad(11.0)
    r_alpha = -P.reward.w_alpha * ((alpha - deg2rad(11.0)) / deg2rad(3.0))^2;
elseif alpha < deg2rad(-4.0)
    r_alpha = -P.reward.w_alpha * ((deg2rad(-4.0) - alpha) / deg2rad(3.0))^2;
end

%% 3. Control Action Effort and Smoothness Costs
% Normalized elevator and throttle deflections
de_norm = de / P.elevator_max;
dT_norm = dT;

r_ctrl_effort = -P.reward.w_elevator * (de_norm^2) - P.reward.w_throttle * (dT_norm^2);

% Action rate / smoothness penalty
delta_de = (de - de_prev) / P.elevator_max;
delta_dT = (dT - dT_prev);
r_smoothness = -P.reward.w_delta_u * (delta_de^2 + delta_dT^2);

%% 4. Forward Progress Incentive (prevents climbing or hovering forever)
ground_vx = V * cos(gamma);
r_progress = P.reward.w_progress * (ground_vx * P.dt);

%% 5. Terminal Touchdown & Envelope Violation Rewards
r_terminal = 0.0;

if landing_status.is_terminal
    switch landing_status.status
        case 'SOFT_TOUCHDOWN'
            % High bonus plus sink-rate accuracy bonus
            sink_bonus = max(0.0, (1.0 - sink_rate / P.td_sink_rate_soft) * 200.0);
            pos_bonus  = max(0.0, (1.0 - abs(x - P.touchdown_aim_x) / 300.0) * 100.0);
            r_terminal = P.reward.r_touchdown_soft + sink_bonus + pos_bonus;

        case 'ACCEPTABLE_TOUCHDOWN'
            r_terminal = P.reward.r_touchdown_soft * 0.65;

        case 'HARD_LANDING'
            r_terminal = P.reward.r_touchdown_hard;

        case {'CRASH_HIGH_SINK_RATE', 'CRASH_SHORT_OF_RUNWAY', 'RUNWAY_OVERRUN', ...
              'NOSE_GEAR_COLLAPSE', 'TAIL_STRIKE', 'STALL_ON_TOUCHDOWN'}
            r_terminal = P.reward.r_crash;

        case {'AERODYNAMIC_STALL', 'LOSS_OF_ATTITUDE_CONTROL', 'MISSED_APPROACH_OVERSHOOT'}
            r_terminal = P.reward.r_crash;

        case 'DEPARTED_GLIDESLOPE_CORRIDOR'
            r_terminal = P.reward.r_out_of_bounds;

        otherwise
            r_terminal = 0.0;
    end
end

%% 6. Total Reward Summation
r_step = r_glideslope + r_airspeed + r_sink + r_pitch + r_alpha + ...
         r_ctrl_effort + r_smoothness + r_progress;

r = r_step + r_terminal;

%% 7. Package Diagnostic Breakdown
breakdown.total       = r;
breakdown.glideslope  = r_glideslope;
breakdown.airspeed    = r_airspeed;
breakdown.sink        = r_sink;
breakdown.pitch       = r_pitch;
breakdown.alpha       = r_alpha;
breakdown.ctrl_effort = r_ctrl_effort;
breakdown.smoothness  = r_smoothness;
breakdown.progress    = r_progress;
breakdown.terminal    = r_terminal;

end
