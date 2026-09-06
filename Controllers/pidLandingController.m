function [controls, ctrl_state, debug_info] = pidLandingController(cmd_or_state, meas, ref, P, trim, ctrl_state)
% PIDLANDINGCONTROLLER Classical multi-loop landing flight control system.
%
% Architecture:
%   1. Inner-loop: Pitch attitude PID controller with pitch-rate damping and anti-windup.
%   2. Outer-loop: Altitude/Glide-slope guidance computing pitch command theta_cmd.
%   3. Flare law:  Exponential sink rate regulation below flare altitude (h <= h_flare).
%   4. Autothrottle: Airspeed error PI controller with feedforward trim throttle.
%
% Subcommands:
%   ctrl_state = pidLandingController('reset', P, trim)
%   [controls, ctrl_state, debug_info] = pidLandingController('step', meas, ref, P, trim, ctrl_state)
%
% Inputs:
%   meas       - Measured aircraft states [x; h; V; gamma; theta; q] or struct
%   ref        - Reference struct from landingScenario('reference', x, h, P)
%   P          - Aircraft parameters struct from aircraftParameters()
%   trim       - Trim conditions struct from initializeAircraft()
%   ctrl_state - Controller internal memory struct (integrators, filters)
%
% Outputs:
%   controls   - [delta_e; delta_T] Actuator command vector
%   ctrl_state - Updated controller internal memory
%   debug_info - Diagnostic telemetry struct

%% Command Dispatcher
if ischar(cmd_or_state) || isstring(cmd_or_state)
    switch lower(char(cmd_or_state))
        case 'reset'
            controls = resetController(meas, ref); % meas=P, ref=trim
            return;
        case 'step'
            % standard step execution below
        otherwise
            error('pidLandingController:UnknownCommand', 'Unknown command: %s', cmd_or_state);
    end
end

if nargin < 4 || isempty(P)
    P = aircraftParameters();
end
if nargin < 5 || isempty(trim)
    [~, trim] = initializeAircraft(P);
end
if nargin < 6 || isempty(ctrl_state)
    ctrl_state = resetController(P, trim);
end

%% 1. Unpack Measurements
if isstruct(meas)
    x     = meas.x;
    h     = meas.h;
    V     = meas.V;
    gamma = meas.gamma;
    theta = meas.theta;
    q     = meas.q;
    hdot  = meas.hdot;
else
    x     = meas(1);
    h     = meas(2);
    V     = meas(3);
    gamma = meas(4);
    theta = meas(5);
    q     = meas(6);
    hdot  = V * sin(gamma);
end

%% 2. Outer-Loop Guidance (Altitude / Glide-Slope Tracking -> theta_cmd)
e_h    = ref.h_ref - h;
e_hdot = ref.hdot_ref - hdot;

if h > P.h_flare
    % APPROACH GLIDE-SLOPE TRACKING
    % Pitch command = trim pitch + proportional altitude correction + damping
    theta_cmd_raw = trim.theta + P.pid.Kp_h * e_h + P.pid.Kd_h * e_hdot;
    theta_cmd = min(P.pid.theta_cmd_max, max(P.pid.theta_cmd_min, theta_cmd_raw));
else
    % FLARE GUIDANCE (Sink-rate regulation toward -0.5 m/s)
    flare_ratio = max(0.0, min(1.0, h / P.h_flare));
    
    % As altitude drops below flare altitude, blend into target touchdown pitch
    theta_flare_cmd = P.pid.theta_flare_bias + P.pid.Kp_flare * (ref.hdot_ref - hdot);
    
    % Smooth blend from glide-slope command into flare command
    theta_gs_cmd = trim.theta + P.pid.Kp_h * e_h + P.pid.Kd_h * e_hdot;
    theta_cmd_raw = (1.0 - flare_ratio) * theta_flare_cmd + flare_ratio * theta_gs_cmd;
    theta_cmd = min(P.pid.theta_cmd_max, max(deg2rad(-2.0), theta_cmd_raw));
end

%% 3. Inner-Loop Pitch Attitude PID Controller (theta_cmd -> delta_e)
e_theta = theta_cmd - theta;

% Derivative filtering for pitch rate
% Filtered pitch rate deriv: dq/dt_filt = N * (q - q_filt)
q_deriv = P.pid.N_theta * (q - ctrl_state.q_filt);
ctrl_state.q_filt = ctrl_state.q_filt + P.dt * q_deriv;

% Proportional, Integral, Derivative components
P_elev = P.pid.Kp_theta * e_theta;
D_elev = -P.pid.Kd_theta * q; % Pitch rate damping feedback

% Tentative unsaturated elevator command (negative sign for longitudinal convention)
delta_e_unsat = trim.delta_e - (P_elev + P.pid.Ki_theta * ctrl_state.int_theta + D_elev);

% Saturation
delta_e = min(P.elevator_max, max(P.elevator_min, delta_e_unsat));

% Conditional Anti-Windup Clamping for Pitch Integrator
% Only integrate if not saturated OR if error drives output away from saturation
is_sat_elev = (delta_e ~= delta_e_unsat);
if ~is_sat_elev || (sign(e_theta) ~= sign(delta_e_unsat - delta_e))
    ctrl_state.int_theta = ctrl_state.int_theta + e_theta * P.dt;
    % Clamp integrator bound
    ctrl_state.int_theta = min(deg2rad(15), max(deg2rad(-15), ctrl_state.int_theta));
end

%% 4. Airspeed Autothrottle Controller (V_ref -> delta_T)
e_V = ref.V_ref - V;

% Derivative filtering for airspeed acceleration
accel_raw = (V - ctrl_state.V_prev) / P.dt;
accel_filt = ctrl_state.accel_filt + P.dt * P.pid.N_V * (accel_raw - ctrl_state.accel_filt);
ctrl_state.accel_filt = accel_filt;
ctrl_state.V_prev = V;

% Autothrottle PI + D
P_thr = P.pid.Kp_V * e_V;
D_thr = -P.pid.Kd_V * accel_filt;

delta_T_unsat = trim.delta_T + P_thr + P.pid.Ki_V * ctrl_state.int_V + D_thr;

% Throttle saturation [0, 1]
delta_T = min(P.throttle_max, max(P.throttle_min, delta_T_unsat));

% Throttle Anti-Windup Clamping
is_sat_thr = (delta_T ~= delta_T_unsat);
if ~is_sat_thr || (sign(e_V) ~= sign(delta_T_unsat - delta_T))
    ctrl_state.int_V = ctrl_state.int_V + e_V * P.dt;
    ctrl_state.int_V = min(5.0, max(-5.0, ctrl_state.int_V));
end

%% 5. Rate Limiting (Actuator Slew Limits)
delta_e_rate = (delta_e - ctrl_state.prev_delta_e) / P.dt;
delta_e_rate_lim = min(P.elevator_rate_max, max(-P.elevator_rate_max, delta_e_rate));
delta_e = ctrl_state.prev_delta_e + delta_e_rate_lim * P.dt;

delta_T_rate = (delta_T - ctrl_state.prev_delta_T) / P.dt;
delta_T_rate_lim = min(P.throttle_rate_max, max(-P.throttle_rate_max, delta_T_rate));
delta_T = ctrl_state.prev_delta_T + delta_T_rate_lim * P.dt;

% Update stored actuator states
ctrl_state.prev_delta_e = delta_e;
ctrl_state.prev_delta_T = delta_T;

controls = [delta_e; delta_T];

%% 6. Diagnostic Information
debug_info.theta_cmd   = theta_cmd;
debug_info.e_theta     = e_theta;
debug_info.e_h         = e_h;
debug_info.e_V         = e_V;
debug_info.e_hdot      = e_hdot;
debug_info.delta_e     = delta_e;
debug_info.delta_T     = delta_T;
debug_info.int_theta   = ctrl_state.int_theta;
debug_info.int_V       = ctrl_state.int_V;

end

%% Helper: Reset Controller State
function state = resetController(P, trim)
if nargin < 1 || isempty(P), P = aircraftParameters(); end
if nargin < 2 || isempty(trim), [~, trim] = initializeAircraft(P); end

state.int_theta    = 0.0;
state.int_V        = 0.0;
state.q_filt       = 0.0;
state.accel_filt   = 0.0;
state.V_prev       = P.V_app;
state.prev_delta_e = trim.delta_e;
state.prev_delta_T = trim.delta_T;
end
