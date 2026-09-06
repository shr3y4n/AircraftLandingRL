function m = metrics(flight_log, P)
% METRICS Computes research-grade quantitative landing and flight performance metrics.
%
% Evaluates trajectory tracking, landing accuracy, touchdown safety criteria,
% actuator control effort, control smoothness, and flight envelope safety margins.
%
% Inputs:
%   flight_log - Time-series flight log struct from simulatePID or evaluateAgent
%   P          - Aircraft parameters struct (optional)
%
% Outputs:
%   m - Struct containing all quantitative scalar metrics

if nargin < 2 || isempty(P)
    P = aircraftParameters();
end

time = flight_log.time;
N = length(time);
dt = P.dt;

%% 1. Trajectory Tracking Errors
eh = flight_log.h_ref - flight_log.h;
eV = flight_log.V_ref - flight_log.V;

m.rms_glideslope_error = sqrt(mean(eh.^2));
m.max_glideslope_error = max(abs(eh));
m.rms_airspeed_error   = sqrt(mean(eV.^2));
m.max_airspeed_error   = max(abs(eV));

%% 2. Flight Envelope and Safety Margins
m.min_altitude = min(flight_log.h);
m.max_alpha    = max(flight_log.alpha);
m.min_alpha    = min(flight_log.alpha);
m.max_pitch    = max(flight_log.theta);
m.min_pitch    = min(flight_log.theta);
m.max_sink_rate = max(flight_log.sink_rate);

% Stall detection: check if angle of attack exceeded stall limit
m.stall_occurred = any(flight_log.alpha >= P.alpha_stall_pos) || any(flight_log.V < P.V_stall * 0.90);

%% 3. Control Actuator Effort and Smoothness
de = flight_log.delta_e;
dT = flight_log.delta_T;

m.max_elevator_deflection = max(abs(de));
m.max_throttle_fraction   = max(dT);

% Control effort: integral(u^2 dt)
m.elevator_effort = sum(de.^2) * dt;
m.throttle_effort = sum(dT.^2) * dt;
m.total_control_effort = m.elevator_effort + m.throttle_effort;

% Control smoothness: integral((du/dt)^2 dt)
if N > 1
    dde_dt = diff(de) / dt;
    ddT_dt = diff(dT) / dt;
    m.elevator_smoothness = sum(dde_dt.^2) * dt;
    m.throttle_smoothness = sum(ddT_dt.^2) * dt;
    m.total_control_smoothness = m.elevator_smoothness + m.throttle_smoothness;
else
    m.elevator_smoothness = 0.0;
    m.throttle_smoothness = 0.0;
    m.total_control_smoothness = 0.0;
end

%% 4. Touchdown Performance
m.time_to_touchdown = time(end);
m.touchdown_x       = flight_log.x(end);
m.touchdown_h       = flight_log.h(end);
m.touchdown_V       = flight_log.V(end);
m.touchdown_sink    = flight_log.sink_rate(end);
m.touchdown_pitch   = flight_log.theta(end);

m.touchdown_pos_error   = m.touchdown_x - P.touchdown_aim_x;
m.touchdown_speed_error = m.touchdown_V - P.V_td;
m.touchdown_pitch_error = m.touchdown_pitch - P.theta_flare_tgt;

%% 5. Mission Outcome Classification
m.status     = flight_log.status;
m.is_success = flight_log.is_success;
m.is_crash   = contains(upper(m.status), 'CRASH') || contains(upper(m.status), 'COLLAPSE') || ...
               contains(upper(m.status), 'STRIKE') || contains(upper(m.status), 'STALL');

% Total episode reward (for RL evaluation)
if isfield(flight_log, 'reward') && ~isempty(flight_log.reward)
    m.total_reward = sum(flight_log.reward);
else
    m.total_reward = NaN;
end

end
