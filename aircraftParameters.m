function P = aircraftParameters()
% AIRCRAFTPARAMETERS Single source of truth for aircraft simulation & control.
%
% Defines realistic physical, aerodynamic, propulsion, control, sensor,
% landing scenario, PID gain, and reinforcement learning parameters for a
% small fixed-wing utility/research aircraft (Navion / Cessna 172 class).
%
% All units are standard SI (m, kg, s, N, rad, rad/s) unless explicitly noted.
%
% Outputs:
%   P - Struct containing all parameter groups.

%% 1. Aircraft Mass and Inertia
P.mass = 1100.0;            % Aircraft total mass [kg]
P.Iyy  = 1750.0;            % Pitch moment of inertia [kg*m^2]
P.cg_arm = 0.0;             % Longitudinal CG offset relative to reference datum [m]

%% 2. Aircraft Geometry
P.S  = 16.2;                % Wing planform reference area [m^2]
P.b  = 10.0;                % Wingspan [m]
P.c  = 1.62;                % Mean aerodynamic chord (MAC) [m]
P.AR = P.b^2 / P.S;         % Wing aspect ratio [-] (~6.17)

%% 3. Atmospheric Environment & Constants
P.rho0 = 1.225;             % Sea-level standard air density [kg/m^3]
P.g    = 9.80665;           % Acceleration due to gravity [m/s^2]

%% 4. Aerodynamic Coefficients & Stability Derivatives
% Non-dimensional longitudinal stability derivatives (body/stability axis)
P.CL0       = 0.28;         % Zero-angle-of-attack lift coefficient [-]
P.CLalpha   = 4.58;         % Lift curve slope [1/rad] (~0.080 1/deg)
P.CLq       = 3.80;         % Pitch rate lift derivative [1/rad]
P.CLdelta_e = 0.36;         % Elevator lift control derivative [1/rad]

P.CD0       = 0.028;        % Parasite (zero-lift) drag coefficient [-]
P.e         = 0.78;         % Oswald efficiency factor [-]
P.K         = 1.0 / (pi * P.e * P.AR); % Induced drag factor [-] (~0.0662)
P.CDdelta_e = 0.04;         % Elevator drag increment factor [1/rad^2]

P.Cm0       = 0.00;         % Pitching moment coefficient at alpha = 0 [-]
P.Cmalpha   = -0.72;        % Static longitudinal stability derivative [1/rad] (negative = stable)
P.Cmq       = -9.00;        % Pitch damping derivative [1/rad] (negative = damped)
P.Cmdelta_e = -1.15;        % Elevator pitching moment control power [1/rad] (negative = pitch down for pos de)

% Stall and High Angle-of-Attack Aerodynamics
P.alpha_stall_pos  = deg2rad(14.0);   % Positive stall angle of attack [rad] (~14.0 deg)
P.alpha_stall_neg  = deg2rad(-10.0);  % Negative stall angle of attack [rad] (~-10.0 deg)
P.CL_max           = 1.45;            % Maximum usable lift coefficient [-]
P.stall_transition = deg2rad(2.0);    % Sigmoid transition width for stall model [rad]

% Ground Effect (Wieselsberger / McCormick formula for landing flare)
P.ground_effect_enabled = true;

%% 5. Propulsion System
P.maxThrust   = 3800.0;     % Maximum installed engine thrust [N]
P.thrust_cant = deg2rad(0); % Engine thrust cant angle relative to waterline [rad]
P.tau_thrust  = 0.35;       % Engine spool-up first-order lag time constant [s]

%% 6. Control Effector Bounds & Rates
P.elevator_min = deg2rad(-25.0); % Minimum elevator deflection (trailing-edge down) [rad]
P.elevator_max = deg2rad(15.0);  % Maximum elevator deflection (trailing-edge up) [rad]
P.elevator_rate_max = deg2rad(40.0); % Maximum elevator angular slew rate [rad/s]

P.throttle_min = 0.0;            % Minimum throttle fraction [-] (idle)
P.throttle_max = 1.0;            % Maximum throttle fraction [-] (full thrust)
P.throttle_rate_max = 0.50;      % Maximum throttle slew rate [1/s]

%% 7. Flight Envelope Operational Limits
P.V_min        = 22.0;           % Low airspeed stall warning threshold [m/s] (~42.7 kt)
P.V_max        = 55.0;           % Maximum operational airspeed [m/s] (~107 kt)
P.alpha_min    = deg2rad(-6.0);  % Minimum allowable angle of attack [rad]
P.alpha_max    = deg2rad(16.0);  % Maximum allowable angle of attack [rad]
P.theta_min    = deg2rad(-15.0); % Minimum allowable pitch attitude [rad]
P.theta_max    = deg2rad(20.0);  % Maximum allowable pitch attitude [rad]
P.sink_rate_max = 6.0;           % Maximum survivable vertical descent rate [m/s]

%% 8. Landing Scenario & Runway Geometry
P.runway_x        = 0.0;          % Runway physical threshold x position [m]
P.runway_h        = 0.0;          % Runway surface elevation [m]
P.runway_length   = 1200.0;       % Total usable runway length [m]
P.touchdown_aim_x = 300.0;        % Target touchdown aim point past threshold [m]

% Approach Reference Trajectory
P.x0        = -3000.0;            % Initial longitudinal distance from threshold [m]
P.gamma_des = deg2rad(-3.0);      % Standard ICAO 3-degree glide-slope angle [rad] (~-0.05236 rad)
% Desired glide-slope starts at altitude h0 aimed toward touchdown point
P.h0        = -(P.x0 - P.touchdown_aim_x) * tan(-P.gamma_des); % Nominal initial altitude (~172.9 m)

% Reference Speeds
P.V_app     = 35.0;               % Nominal approach airspeed [m/s] (~68.0 kt)
P.V_td      = 28.0;               % Target touchdown airspeed [m/s] (~54.4 kt)
P.V_stall   = sqrt((2 * P.mass * P.g) / (P.rho0 * P.S * P.CL_max)); % Nominal stall speed [m/s] (~24.5 m/s)

% Flare Phase Parameters
P.h_flare         = 12.0;         % Flare initiation altitude above runway [m]
P.hdot_flare_td   = -0.50;        % Target vertical sink rate at wheel touchdown [m/s]
P.theta_flare_tgt = deg2rad(3.5); % Target pitch attitude during touchdown flare [rad]

% Quantitative Touchdown Classification Thresholds
P.td_sink_rate_soft  = 1.0;       % Maximum sink rate for soft/normal touchdown [m/s]
P.td_sink_rate_max   = 1.8;       % Maximum acceptable sink rate for safe landing [m/s]
P.td_sink_rate_crash = 3.5;       % Sink rate threshold for structural collapse / crash [m/s]
P.td_x_min           = 0.0;       % Minimum acceptable touchdown x (on runway) [m]
P.td_x_max           = 650.0;     % Maximum acceptable touchdown x [m]
P.td_theta_min       = deg2rad(0.5); % Minimum acceptable touchdown pitch (prevents nose gear first) [rad]
P.td_theta_max       = deg2rad(8.5); % Maximum acceptable touchdown pitch (prevents tailstrike) [rad]

%% 9. Simulation & Numerical Integration
P.dt        = 0.02;               % Simulation fundamental integration timestep [s] (50 Hz)
P.t_max     = 120.0;              % Maximum simulation duration [s]
P.integrator = 'rk4';             % Default numerical integrator: 'rk4' or 'ode45'

%% 10. Classical PID Baseline Controller Parameters
% Inner-loop Pitch Attitude Controller
P.pid.Kp_theta = 2.40;            % Pitch error proportional gain [rad_e/rad_err]
P.pid.Ki_theta = 0.35;            % Pitch error integral gain [rad_e/(rad_err*s)]
P.pid.Kd_theta = 0.65;            % Pitch rate damping gain [rad_e/(rad/s)]
P.pid.N_theta  = 30.0;            % Pitch derivative filter coefficient [rad/s]

% Outer-loop Altitude / Glide-slope Tracking
P.pid.Kp_h = 0.022;               % Altitude tracking proportional gain [rad_cmd/m]
P.pid.Kd_h = 0.035;               % Altitude rate / sink tracking gain [rad_cmd/(m/s)]
P.pid.theta_cmd_min = deg2rad(-8.0);  % Minimum pitch command during approach [rad]
P.pid.theta_cmd_max = deg2rad(10.0);  % Maximum pitch command during approach [rad]

% Flare Guidance Law Gains (Active when h <= h_flare)
P.pid.Kp_flare = 0.055;           % Flare sink rate proportional gain [rad_cmd/(m/s)]
P.pid.Kd_flare = 0.075;           % Flare vertical acceleration damping gain
P.pid.theta_flare_bias = deg2rad(3.2); % Flare trim bias attitude [rad]

% Airspeed Throttle Controller (Autothrottle)
P.pid.Kp_V = 0.085;               % Airspeed error proportional gain [1/(m/s)]
P.pid.Ki_V = 0.018;               % Airspeed error integral gain [1/((m/s)*s)]
P.pid.Kd_V = 0.020;               % Airspeed acceleration derivative gain [1/((m/s^2))]
P.pid.N_V  = 15.0;                % Airspeed derivative filter coefficient [rad/s]

%% 11. Sensor Noise & Measurement Model
P.sensor.enabled        = false;  % Enable measurement corruption if true
P.sensor.sigma_V        = 0.25;   % Airspeed measurement Gaussian noise std [m/s]
P.sensor.sigma_h        = 0.50;   % Altimeter measurement Gaussian noise std [m]
P.sensor.sigma_theta    = deg2rad(0.15); % Pitch angle measurement noise std [rad]
P.sensor.sigma_q        = deg2rad(0.20); % Gyro pitch rate noise std [rad/s]
P.sensor.sigma_hdot     = 0.10;   % Variometer sink rate noise std [m/s]
P.sensor.sigma_x        = 1.00;   % DME / GPS distance measurement noise std [m]

%% 12. Reinforcement Learning (RL) Configuration & Reward Weights
% Observation normalizers
P.rl.obs_norm.x    = 3000.0;      % Distance scale [m]
P.rl.obs_norm.h    = 200.0;       % Altitude scale [m]
P.rl.obs_norm.eh   = 50.0;        % Altitude error scale [m]
P.rl.obs_norm.eV   = 10.0;        % Airspeed error scale [m/s]
P.rl.obs_norm.hdot = 10.0;        % Vertical speed scale [m/s]

% Reward Function Weights
P.reward.w_eh       = 0.025;      % Weight on glide-slope altitude tracking error
P.reward.w_eV       = 0.070;      % Weight on airspeed tracking error
P.reward.w_hdot     = 0.120;      % Weight on excessive sink rate
P.reward.w_theta    = 0.180;      % Weight on pitch attitude deviation
P.reward.w_alpha    = 0.250;      % Weight on excessive angle of attack
P.reward.w_elevator = 0.030;      % Weight on elevator control deflection effort
P.reward.w_throttle = 0.015;      % Weight on throttle control effort
P.reward.w_delta_u  = 0.080;      % Weight on control input rate / action jerk
P.reward.w_progress = 0.004;      % Bonus weight on forward progress toward runway

% Terminal Rewards and Penalties
P.reward.r_touchdown_soft = 1000.0; % High bonus for soft touchdown in zone
P.reward.r_touchdown_hard = -200.0; % Penalty for hard touchdown exceeding sink rate
P.reward.r_crash          = -1000.0;% Heavy penalty for stall, dive, or crash
P.reward.r_out_of_bounds  = -500.0; % Penalty for straying outside approach corridor

end