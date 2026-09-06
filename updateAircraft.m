function [next_state, info] = updateAircraft(state, controls, dt, P, wind_input)
% UPDATEAIRCRAFT Advances the aircraft state by timestep dt using Runge-Kutta 4 (RK4).
%
% Integrates the continuous-time nonlinear equations of motion defined in
% aircraftDynamics.m. Supports both struct and column-vector state interfaces.
%
% Inputs:
%   state      - Current aircraft state (struct or [6x1] vector)
%   controls   - Control input (struct or [2x1] vector: [delta_e; delta_T])
%   dt         - Timestep for integration [s] (defaults to P.dt)
%   P          - Aircraft parameters struct from aircraftParameters()
%   wind_input - Wind velocity/config (optional)
%
% Outputs:
%   next_state - Updated aircraft state at time t + dt
%   info       - Diagnostic telemetry struct at the end of the step

if nargin < 3 || isempty(dt)
    dt = P.dt;
end
if nargin < 4 || isempty(P)
    P = aircraftParameters();
end
if nargin < 5
    wind_input = [];
end

is_struct = isstruct(state);

%% 1. Convert State to Vector Form
if is_struct
    x_vec = [state.x; state.h; state.V; state.gamma; state.theta; state.q];
else
    x_vec = state(:);
end

%% 2. Convert Controls to Vector Form
if isstruct(controls)
    u_vec = [controls.elevator; controls.throttle];
else
    u_vec = controls(:);
end

%% 3. Numerical Integration (4th-Order Runge-Kutta)
t0 = 0.0; % Dynamics are autonomous or explicitly driven by time-dependent wind

% k1
[k1, info1] = aircraftDynamics(t0, x_vec, u_vec, P, wind_input);

% k2
x_k2 = x_vec + 0.5 * dt * k1;
[k2, ~]     = aircraftDynamics(t0 + 0.5 * dt, x_k2, u_vec, P, wind_input);

% k3
x_k3 = x_vec + 0.5 * dt * k2;
[k3, ~]     = aircraftDynamics(t0 + 0.5 * dt, x_k3, u_vec, P, wind_input);

% k4
x_k4 = x_vec + dt * k3;
[k4, ~]     = aircraftDynamics(t0 + dt, x_k4, u_vec, P, wind_input);

% Next state vector
next_vec = x_vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

%% 4. Ground Contact and Physical Safety Envelopes
% If aircraft has touched down (h <= 0), prevent ground penetration
if next_vec(2) <= 0.0
    next_vec(2) = 0.0;
    % When wheels are firmly on ground, flight-path angle gamma aligns with ground
    if next_vec(4) < 0.0
        next_vec(4) = 0.0;
    end
end

% Angle wrapping for pitch angle
if next_vec(5) > pi,  next_vec(5) = next_vec(5) - 2.0 * pi; end
if next_vec(5) < -pi, next_vec(5) = next_vec(5) + 2.0 * pi; end

% Evaluate final step telemetry
[~, info] = aircraftDynamics(t0 + dt, next_vec, u_vec, P, wind_input);

%% 5. Return in Matching Format
if is_struct
    next_state.x        = next_vec(1);
    next_state.h        = next_vec(2);
    next_state.V        = next_vec(3);
    next_state.gamma    = next_vec(4);
    next_state.theta    = next_vec(5);
    next_state.q        = next_vec(6);
    next_state.elevator = u_vec(1);
    next_state.throttle = u_vec(2);
else
    next_state = next_vec;
end

end
