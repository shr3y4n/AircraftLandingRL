function [xdot, info] = aircraftDynamics(t, state, controls, P, wind_input)
% AIRCRAFTDYNAMICS Computes time derivatives of nonlinear longitudinal aircraft states.
%
% State vector definition:
%   x(1) = x     - Horizontal position along runway axis [m]
%   x(2) = h     - Altitude above runway surface [m]
%   x(3) = V     - True airspeed (relative to air mass) [m/s]
%   x(4) = gamma - Air-relative flight-path angle [rad]
%   x(5) = theta - Pitch attitude angle [rad]
%   x(6) = q     - Pitch rate [rad/s]
%
% Control input vector:
%   controls(1) = delta_e  - Elevator deflection angle [rad]
%   controls(2) = delta_T  - Throttle fraction [0 to 1]
%
% Environmental input:
%   wind_input - [wx; wh] wind velocity vector or wind config struct (optional)
%
% Outputs:
%   xdot - State derivative vector [dx/dt; dh/dt; dV/dt; dgamma/dt; dtheta/dt; dq/dt]
%   info - Struct containing aerodynamic forces, moments, coefficients, and flight data

%% 1. Unpack State Vector
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

% Prevent singularity at near-zero or negative airspeed
V = max(5.0, V);

%% 2. Unpack and Saturate Controls
if isstruct(controls)
    delta_e = controls.elevator;
    delta_T = controls.throttle;
else
    delta_e = controls(1);
    delta_T = controls(2);
end

% Physical actuator saturation
delta_e = min(P.elevator_max, max(P.elevator_min, delta_e));
delta_T = min(P.throttle_max, max(P.throttle_min, delta_T));

%% 3. Atmospheric Wind Resolution
wx     = 0.0;
wh     = 0.0;
dwx_dt = 0.0;
dwh_dt = 0.0;

if nargin >= 5 && ~isempty(wind_input)
    if isstruct(wind_input)
        [w_vel, w_dot] = windModel(t, [x; h; V; gamma; theta; q], wind_input);
        wx     = w_vel(1);
        wh     = w_vel(2);
        dwx_dt = w_dot(1);
        dwh_dt = w_dot(2);
    elseif isnumeric(wind_input)
        wx = wind_input(1);
        if length(wind_input) >= 2, wh = wind_input(2); end
        if length(wind_input) >= 4
            dwx_dt = wind_input(3);
            dwh_dt = wind_input(4);
        end
    end
end

%% 4. Kinematics & Aerodynamics
% Angle of attack (in air-mass frame)
alpha = theta - gamma;

% Dynamic pressure
qbar = 0.5 * P.rho0 * V^2;

% Ground effect correction on induced drag (near ground h < wingspan b)
K_eff = P.K;
if isfield(P, 'ground_effect_enabled') && P.ground_effect_enabled
    h_eff = max(0.5, h);
    h_b_ratio = h_eff / P.b;
    sigma_ge = (16.0 * h_b_ratio^2) / (1.0 + 16.0 * h_b_ratio^2);
    sigma_ge = min(1.0, max(0.20, sigma_ge));
    K_eff = P.K * sigma_ge;
end

% Non-dimensional damping non-dimensionalizing factor
q_hat = (P.c / (2.0 * V)) * q;

% Linear pre-stall coefficients
CL_lin = P.CL0 + P.CLalpha * alpha + P.CLdelta_e * delta_e + P.CLq * q_hat;

% Nonlinear stall blending model (Smoothstep C^1 Hermite transition)
% Strictly 0 in normal flight envelope (|alpha| < alpha_stall - transition),
% smoothly blends to 1.0 when |alpha| > alpha_stall + transition.
da = P.stall_transition;
f_pos = 0.0;
if alpha > (P.alpha_stall_pos + da)
    f_pos = 1.0;
elseif alpha > (P.alpha_stall_pos - da)
    s = (alpha - (P.alpha_stall_pos - da)) / (2.0 * da);
    f_pos = 3.0 * s^2 - 2.0 * s^3;
end

f_neg = 0.0;
if alpha < (P.alpha_stall_neg - da)
    f_neg = 1.0;
elseif alpha < (P.alpha_stall_neg + da)
    s = ((P.alpha_stall_neg + da) - alpha) / (2.0 * da);
    f_neg = 3.0 * s^2 - 2.0 * s^3;
end
f_stall = min(1.0, f_pos + f_neg);

% Separated flow lift coefficient (flat-plate approximation)
CL_sep = 2.0 * sign(alpha) * (sin(alpha))^2 * cos(alpha);

% Blended lift coefficient
CL = (1.0 - f_stall) * CL_lin + f_stall * CL_sep;

% Drag coefficient (parasite + induced + control deflection + post-stall rise)
CD = P.CD0 + K_eff * CL^2 + P.CDdelta_e * (delta_e^2) + f_stall * (2.0 * (sin(abs(alpha)))^2);

% Pitching moment coefficient
Cm = P.Cm0 + P.Cmalpha * alpha + P.Cmdelta_e * delta_e + P.Cmq * q_hat;

% Post-stall nose-down pitching moment bias (aerodynamic recovery characteristic)
if f_stall > 0.01
    Cm = Cm - 0.5 * f_stall * sign(alpha);
end

%% 5. Dimensional Forces and Moments
Lift   = qbar * P.S * CL;
Drag   = qbar * P.S * CD;
Thrust = delta_T * P.maxThrust;
M_pitch = qbar * P.S * P.c * Cm;

%% 6. Equations of Motion
% Kinematics (inertial ground frame position)
xdot_pos = V * cos(gamma) + wx;
hdot_pos = V * sin(gamma) + wh;

% Translational acceleration (wind-axis / velocity frame with atmospheric wind gradient)
% m * dV/dt = T*cos(alpha) - Drag - m*g*sin(gamma) - m*(dwx_dt*cos(gamma) + dwh_dt*sin(gamma))
Vdot = (Thrust * cos(alpha) - Drag) / P.mass - P.g * sin(gamma) ...
       - (dwx_dt * cos(gamma) + dwh_dt * sin(gamma));

% Flight-path angular rate
% m*V * dgamma/dt = T*sin(alpha) + Lift - m*g*cos(gamma) + m*(dwx_dt*sin(gamma) - dwh_dt*cos(gamma))
gammadot = (Thrust * sin(alpha) + Lift) / (P.mass * V) - (P.g * cos(gamma)) / V ...
           + (dwx_dt * sin(gamma) - dwh_dt * cos(gamma)) / V;

% Pitch attitude kinematics and pitch rate dynamics
thetadot = q;
qdot     = M_pitch / P.Iyy;

%% 7. Assemble State Derivative Vector
xdot = [xdot_pos; hdot_pos; Vdot; gammadot; thetadot; qdot];

%% 8. Telemetry and Diagnostic Info
if nargout >= 2
    info.x          = x;
    info.h          = h;
    info.V          = V;
    info.gamma      = gamma;
    info.theta      = theta;
    info.q          = q;
    info.alpha      = alpha;
    info.Lift       = Lift;
    info.Drag       = Drag;
    info.Thrust     = Thrust;
    info.M_pitch    = M_pitch;
    info.CL         = CL;
    info.CD         = CD;
    info.Cm         = Cm;
    info.qbar       = qbar;
    info.delta_e    = delta_e;
    info.delta_T    = delta_T;
    info.wx         = wx;
    info.wh         = wh;
    info.ground_speed = sqrt(xdot_pos^2 + hdot_pos^2);
    info.sink_rate  = -hdot_pos; % Sink rate is positive descent [m/s]
    info.is_stalled = (f_stall > 0.5);
end

end