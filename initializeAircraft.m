function [state, trim] = initializeAircraft(P, init_options)
% INITIALIZEAIRCRAFT Computes trim condition and initial state for flight simulation.
%
% Numerically solves the trimmed steady-flight equations:
%   Lift + Thrust*sin(alpha) = mass * g * cos(gamma)
%   Thrust*cos(alpha) = Drag + mass * g * sin(gamma)
%   Pitching Moment M = 0
%
% Inputs:
%   P            - Aircraft parameter struct from aircraftParameters()
%   init_options - Optional struct to override initial conditions:
%                  .x0        - Initial horizontal position [m]
%                  .h0        - Initial altitude [m]
%                  .V0        - Initial true airspeed [m/s]
%                  .gamma0    - Initial flight-path angle [rad]
%                  .theta0    - Initial pitch attitude [rad] (optional)
%                  .q0        - Initial pitch rate [rad/s]
%
% Outputs:
%   state - Struct with initial states: x, h, V, gamma, theta, q, throttle, elevator
%   trim  - Struct with calculated trim parameters: alpha, delta_e, delta_T, Thrust, Drag, Lift

if nargin < 1 || isempty(P)
    P = aircraftParameters();
end
if nargin < 2
    init_options = struct();
end

%% 1. Default Flight Condition
V0     = P.V_app;
gamma0 = P.gamma_des;
x0     = P.x0;
h0     = P.h0;

if isfield(init_options, 'V0'),     V0     = init_options.V0; end
if isfield(init_options, 'gamma0'), gamma0 = init_options.gamma0; end
if isfield(init_options, 'x0'),     x0     = init_options.x0; end
if isfield(init_options, 'h0'),     h0     = init_options.h0; end

%% 2. Iterative Trim Computation
% Dynamic pressure at initial airspeed
qbar = 0.5 * P.rho0 * V0^2;
mg   = P.mass * P.g;

% Ground effect factor at initial altitude
K_eff = P.K;
if isfield(P, 'ground_effect_enabled') && P.ground_effect_enabled
    h_eff = max(0.5, h0);
    h_b_ratio = h_eff / P.b;
    sigma_ge = (16.0 * h_b_ratio^2) / (1.0 + 16.0 * h_b_ratio^2);
    sigma_ge = min(1.0, max(0.20, sigma_ge));
    K_eff = P.K * sigma_ge;
end

% Initial estimate of required lift
L_target = mg * cos(gamma0);
CL_target = L_target / (qbar * P.S);

% Effective lift curve slope accounting for elevator moment trim (Cm = 0)
% delta_e = -(Cm0 + Cmalpha*alpha) / Cmdelta_e
% CL = CL0 + CLalpha*alpha + CLdelta_e*delta_e
% => CL = (CL0 - CLdelta_e*Cm0/Cmdelta_e) + (CLalpha - CLdelta_e*Cmalpha/Cmdelta_e)*alpha
dCL_dalpha_trim = P.CLalpha - (P.CLdelta_e * P.Cmalpha / P.Cmdelta_e);
CL0_trim        = P.CL0 - (P.CLdelta_e * P.Cm0 / P.Cmdelta_e);

alpha_trim = (CL_target - CL0_trim) / dCL_dalpha_trim;
delta_e_trim = -(P.Cm0 + P.Cmalpha * alpha_trim) / P.Cmdelta_e;

% Newton-Raphson refinement (3 iterations for exact nonlinear convergence)
for iter = 1:5
    CL = P.CL0 + P.CLalpha * alpha_trim + P.CLdelta_e * delta_e_trim;
    CD = P.CD0 + K_eff * CL^2 + P.CDdelta_e * delta_e_trim^2;
    Drag = qbar * P.S * CD;
    
    % Thrust from along-path balance
    T_req = (Drag + mg * sin(gamma0)) / cos(alpha_trim);
    T_req = max(0.0, min(P.maxThrust, T_req));
    
    % Normal force residual
    L_req = mg * cos(gamma0) - T_req * sin(alpha_trim);
    CL_req = L_req / (qbar * P.S);
    
    % Residual error and update
    res_CL = CL_req - CL;
    alpha_trim = alpha_trim + res_CL / dCL_dalpha_trim;
    delta_e_trim = -(P.Cm0 + P.Cmalpha * alpha_trim) / P.Cmdelta_e;
end

% Bound controls to physical limits
delta_e_trim = min(P.elevator_max, max(P.elevator_min, delta_e_trim));
delta_T_trim = min(P.throttle_max, max(P.throttle_min, T_req / P.maxThrust));

theta_trim = gamma0 + alpha_trim;

%% 3. Package Trim Struct
trim.alpha   = alpha_trim;
trim.theta   = theta_trim;
trim.delta_e = delta_e_trim;
trim.delta_T = delta_T_trim;
trim.Thrust  = delta_T_trim * P.maxThrust;
trim.Drag    = Drag;
trim.Lift    = qbar * P.S * CL;
trim.CL      = CL;
trim.CD      = CD;

%% 4. Assemble State Struct
state.x        = x0;
state.h        = h0;
state.V        = V0;
state.gamma    = gamma0;
state.theta    = theta_trim;
state.q        = 0.0;
state.throttle = delta_T_trim;
state.elevator = delta_e_trim;

% Allow user overrides for off-nominal or domain-randomized tests
if isfield(init_options, 'theta0'), state.theta    = init_options.theta0; end
if isfield(init_options, 'q0'),     state.q        = init_options.q0;     end
if isfield(init_options, 'delta_e'),state.elevator = init_options.delta_e; end
if isfield(init_options, 'delta_T'),state.throttle = init_options.delta_T; end

end