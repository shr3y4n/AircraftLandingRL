function out = landingScenario(cmd, varargin)
% LANDINGSCENARIO Defines approach reference trajectory and touchdown evaluation.
%
% Sub-functions:
%   ref = landingScenario('reference', x, h, P)
%   [status, is_terminal, details] = landingScenario('evaluate', state, P)
%   corridor = landingScenario('corridor', x, P)

switch lower(cmd)
    case 'reference'
        out = getReference(varargin{:});
    case 'evaluate'
        [out.status, out.is_terminal, out.details] = evaluateStatus(varargin{:});
    case 'corridor'
        out = getCorridor(varargin{:});
    otherwise
        error('landingScenario:UnknownCommand', 'Unknown subcommand: %s', cmd);
end

end

%% Sub-function: Reference Trajectory Generation
function ref = getReference(x, h, P)
% Computes reference altitude, sink rate, airspeed, and flight-path angle.

if nargin < 3 || isempty(P)
    P = aircraftParameters();
end

% Nominal 3-degree glide-slope targeting the touchdown aim point (x = 300 m)
% h_gs(x) = -(x - x_aim) * tan(-gamma_des)
h_gs = -(x - P.touchdown_aim_x) * tan(-P.gamma_des);
h_gs = max(0.0, h_gs);

% Standard glide-slope sink rate [m/s]
hdot_gs = P.V_app * sin(P.gamma_des); % ~ -1.83 m/s

if h > P.h_flare
    % APPROACH PHASE (Above flare altitude)
    ref.phase      = 'approach';
    ref.h_ref      = h_gs;
    ref.hdot_ref   = hdot_gs;
    ref.V_ref      = P.V_app;
    ref.gamma_ref  = P.gamma_des;
    ref.theta_ref  = P.gamma_des + deg2rad(4.2); % Nominal approach deck angle
    ref.flare_prog = 0.0;
else
    % FLARE PHASE (Within flare altitude zone, transitioning toward touchdown)
    ref.phase      = 'flare';
    flare_ratio    = max(0.0, min(1.0, h / P.h_flare));
    ref.flare_prog = 1.0 - flare_ratio;
    
    % Exponential flare sink rate regulation profile:
    % As h approaches 0, sink rate flattens from hdot_gs (-1.83 m/s) to hdot_flare_td (-0.50 m/s)
    ref.hdot_ref   = P.hdot_flare_td + (hdot_gs - P.hdot_flare_td) * flare_ratio;
    
    % Reference altitude during flare smoothly decays to zero
    ref.h_ref      = P.h_flare * (flare_ratio^1.5);
    
    % Reference airspeed bleeds smoothly from approach speed to touchdown target speed
    ref.V_ref      = P.V_td + (P.V_app - P.V_td) * flare_ratio;
    
    % Reference pitch rotates smoothly nose-up to target touchdown flare pitch
    ref.theta_ref  = P.theta_flare_tgt + (P.gamma_des + deg2rad(4.2) - P.theta_flare_tgt) * flare_ratio;
    
    % Flight-path angle in flare
    ref.gamma_ref  = asin(ref.hdot_ref / max(5.0, ref.V_ref));
end

ref.e_h = ref.h_ref - h;

end

%% Sub-function: Touchdown and In-Flight Status Evaluation
function [status, is_terminal, details] = evaluateStatus(state, P)
% Evaluates current state against touchdown and flight envelope boundaries.

if nargin < 2 || isempty(P)
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

alpha = theta - gamma;
sink_rate = -V * sin(gamma); % Descent speed [m/s]

ref = getReference(x, h, P);

details.x         = x;
details.h         = h;
details.V         = V;
details.gamma     = gamma;
details.theta     = theta;
details.alpha     = alpha;
details.sink_rate = sink_rate;
details.ref       = ref;
details.is_success = false;

% 1. CHECK TOUCHDOWN (Altitude reaches or crosses runway ground level)
if h <= 0.05
    is_terminal = true;
    details.touchdown_x = x;
    details.touchdown_sink_rate = sink_rate;
    details.touchdown_speed = V;
    details.touchdown_pitch = theta;
    
    % Verify runway touchdown zone bounds
    in_td_zone = (x >= P.td_x_min) && (x <= P.td_x_max);
    
    if ~in_td_zone
        if x < P.td_x_min
            status = 'CRASH_SHORT_OF_RUNWAY';
        else
            status = 'RUNWAY_OVERRUN';
        end
        return;
    end
    
    % Check pitch attitude limits
    if theta < P.td_theta_min
        status = 'NOSE_GEAR_COLLAPSE';
        return;
    elseif theta > P.td_theta_max
        status = 'TAIL_STRIKE';
        return;
    end
    
    % Check airspeed at touchdown
    if V < P.V_stall * 0.95
        status = 'STALL_ON_TOUCHDOWN';
        return;
    end
    
    % Check sink rate
    if sink_rate <= P.td_sink_rate_soft
        status = 'SOFT_TOUCHDOWN';
        details.is_success = true;
    elseif sink_rate <= P.td_sink_rate_max
        status = 'ACCEPTABLE_TOUCHDOWN';
        details.is_success = true;
    elseif sink_rate <= P.td_sink_rate_crash
        status = 'HARD_LANDING';
        details.is_success = false;
    else
        status = 'CRASH_HIGH_SINK_RATE';
        details.is_success = false;
    end
    return;
end

% 2. CHECK IN-FLIGHT CATASTROPHIC ENVELOPE EXCURSIONS
if x > (P.touchdown_aim_x + 800.0) && (h > 15.0)
    % Overshot runway while still high
    status = 'MISSED_APPROACH_OVERSHOOT';
    is_terminal = true;
    return;
end

if (alpha > P.alpha_max) || (V < P.V_stall * 0.85)
    status = 'AERODYNAMIC_STALL';
    is_terminal = true;
    return;
end

if (theta < P.theta_min) || (theta > P.theta_max)
    status = 'LOSS_OF_ATTITUDE_CONTROL';
    is_terminal = true;
    return;
end

if abs(h - ref.h_ref) > 120.0
    status = 'DEPARTED_GLIDESLOPE_CORRIDOR';
    is_terminal = true;
    return;
end

% Normal ongoing flight
status = 'IN_FLIGHT';
is_terminal = false;

end

%% Sub-function: Approach Corridor Bounds
function corridor = getCorridor(x, P)
% Computes acceptable approach corridor bounds as function of distance x.

if nargin < 2 || isempty(P)
    P = aircraftParameters();
end

ref = getReference(x, 100.0, P);
h_nominal = ref.h_ref;

% Corridor narrows as aircraft approaches the runway
dist_to_td = max(0.0, P.touchdown_aim_x - x);
margin_h   = 15.0 + 0.035 * dist_to_td; % Expands from 15m at touchdown to ~120m at 3km

corridor.x_grid = x;
corridor.h_ref  = h_nominal;
corridor.h_up   = h_nominal + margin_h;
corridor.h_low  = max(0.0, h_nominal - margin_h);

end
