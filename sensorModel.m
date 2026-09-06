function [meas, filter_state] = sensorModel(true_state, P, filter_state)
% SENSORMODEL Simulates realistic sensor measurement noise and filtering.
%
% Accurately separates true physical aircraft states from measured signals.
% Simulates noise characteristics of:
%   - Pitot-static tube (airspeed)
%   - Barometric / Radar altimeter (altitude)
%   - IMU gyroscopes and inclinometers (pitch, pitch rate)
%   - GPS / DME (horizontal distance)
%
% Inputs:
%   true_state   - Vector [6x1] or struct of true physical states
%   P            - Parameters struct from aircraftParameters()
%   filter_state - Previous filter internal state struct (optional)
%
% Outputs:
%   meas         - Struct of measured states with sensor corruption
%   filter_state - Updated filter state struct

if nargin < 2 || isempty(P)
    P = aircraftParameters();
end
if nargin < 3
    filter_state = [];
end

if isstruct(true_state)
    x     = true_state.x;
    h     = true_state.h;
    V     = true_state.V;
    gamma = true_state.gamma;
    theta = true_state.theta;
    q     = true_state.q;
else
    x     = true_state(1);
    h     = true_state(2);
    V     = true_state(3);
    gamma = true_state(4);
    theta = true_state(5);
    q     = true_state(6);
end

if ~isfield(P, 'sensor') || ~isfield(P.sensor, 'enabled') || ~P.sensor.enabled
    % Clean ideal measurements
    meas.x         = x;
    meas.h         = max(0.0, h);
    meas.V         = V;
    meas.gamma     = gamma;
    meas.theta     = theta;
    meas.q         = q;
    meas.hdot      = V * sin(gamma);
    meas.alpha     = theta - gamma;
    return;
end

%% 1. Apply Gaussian Sensor Noise
noise_x     = P.sensor.sigma_x     * randn();
noise_h     = P.sensor.sigma_h     * randn();
noise_V     = P.sensor.sigma_V     * randn();
noise_theta = P.sensor.sigma_theta * randn();
noise_q     = P.sensor.sigma_q     * randn();

raw_x     = x + noise_x;
raw_h     = max(0.0, h + noise_h);
raw_V     = max(1.0, V + noise_V);
raw_theta = theta + noise_theta;
raw_q     = q + noise_q;

%% 2. Measurement Low-Pass Filtering
% Simple 1st-order complementary low-pass filter: y_filt = alpha_f * y_filt + (1 - alpha_f) * y_raw
alpha_f = 0.85; % Filter smoothing factor

if isempty(filter_state)
    filter_state.x     = raw_x;
    filter_state.h     = raw_h;
    filter_state.V     = raw_V;
    filter_state.theta = raw_theta;
    filter_state.q     = raw_q;
    filter_state.hdot  = V * sin(gamma);
else
    filter_state.x     = alpha_f * filter_state.x     + (1.0 - alpha_f) * raw_x;
    filter_state.h     = alpha_f * filter_state.h     + (1.0 - alpha_f) * raw_h;
    filter_state.V     = alpha_f * filter_state.V     + (1.0 - alpha_f) * raw_V;
    filter_state.theta = alpha_f * filter_state.theta + (1.0 - alpha_f) * raw_theta;
    filter_state.q     = alpha_f * filter_state.q     + (1.0 - alpha_f) * raw_q;
    
    % Filtered vertical sink rate from altitude delta
    hdot_est = (raw_h - filter_state.h) / P.dt;
    filter_state.hdot = alpha_f * filter_state.hdot + (1.0 - alpha_f) * hdot_est;
end

meas.x     = filter_state.x;
meas.h     = max(0.0, filter_state.h);
meas.V     = filter_state.V;
meas.theta = filter_state.theta;
meas.q     = filter_state.q;
meas.hdot  = filter_state.hdot;
meas.gamma = asin(max(-0.95, min(0.95, meas.hdot / max(5.0, meas.V))));
meas.alpha = meas.theta - meas.gamma;

end
