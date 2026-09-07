function varargout = runPIDWind(wind_type)
% RUNPIDWIND Evaluates classical PID landing under atmospheric disturbances.
%
% Tests PID approach, flare, and touchdown performance under:
%   - Boundary layer vertical wind shear
%   - Discrete 1-cosine horizontal/vertical wind gusts
%   - Continuous Dryden atmospheric turbulence
%
% Inputs:
%   wind_type - 'shear' (default), 'gust', 'turbulence', or 'combined'
%
% Outputs:
%   flight_log - Time-series flight telemetry struct
%   m          - Quantitative performance metrics struct from metrics.m
%
% Outputs are saved to:
%   Results/Data/pid_wind_results.mat
%   Results/Figures/pid_wind_trajectory.png
%   Results/Figures/pid_wind_telemetry.png

if nargin < 1 || isempty(wind_type), wind_type = 'shear'; end

%% 1. Environment & Path Setup
exp_dir   = fileparts(mfilename('fullpath'));
repo_root = fileparts(exp_dir);
addpath(genpath(repo_root));

data_dir = fullfile(repo_root, 'Results', 'Data');
fig_dir  = fullfile(repo_root, 'Results', 'Figures');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(fig_dir, 'dir'),  mkdir(fig_dir);  end

fprintf('========================================================================\n');
fprintf('  EXPERIMENT: CLASSICAL PID UNDER ATMOSPHERIC DISTURBANCES (%s)\n', upper(wind_type));
fprintf('========================================================================\n');

%% 2. Configure Disturbance
P = aircraftParameters();
wind_cfg = struct();

switch lower(wind_type)
    case 'shear'
        wind_cfg.mode = 'shear';
        wind_cfg.V_headwind = 6.0;   % 6 m/s (~11.7 kt) headwind at altitude
        wind_cfg.z0 = 0.05;          % Aerodynamic roughness length [m]
        desc_str = 'Atmospheric Boundary Layer Wind Shear (6 m/s headwind)';
        
    case 'gust'
        wind_cfg.mode = 'gust';
        wind_cfg.V_headwind = 4.0;
        wind_cfg.gust_Ax    = 5.0;   % +5 m/s horizontal gust amplitude
        wind_cfg.gust_Ah    = -2.5;  % -2.5 m/s downdraft gust amplitude
        wind_cfg.gust_start = 25.0;  % Initiated at t = 25s
        wind_cfg.gust_dur   = 5.0;   % 5-second duration 1-cosine shape
        desc_str = 'Discrete 1-Cosine Wind Gusts (+5 m/s horiz, -2.5 m/s downdraft)';
        
    case 'turbulence'
        wind_cfg.mode = 'turbulence';
        wind_cfg.V_headwind = 5.0;
        wind_cfg.turb_level = 'moderate';
        wind_cfg.seed = 42;
        desc_str = 'Dryden Continuous Atmospheric Turbulence (Moderate)';
        
    case 'combined'
        wind_cfg.mode = 'combined';
        wind_cfg.V_headwind = 6.0;
        wind_cfg.gust_Ax    = 4.0;
        wind_cfg.gust_Ah    = -2.0;
        wind_cfg.gust_start = 20.0;
        wind_cfg.gust_dur   = 6.0;
        wind_cfg.turb_level = 'moderate';
        wind_cfg.seed = 101;
        desc_str = 'Combined Wind Shear, Discrete Gust, and Dryden Turbulence';
        
    otherwise
        error('runPIDWind:UnknownWindType', 'Unknown wind_type: %s', wind_type);
end

fprintf('Disturbance Profile: %s\n', desc_str);
fprintf('Simulating closed-loop approach, flare, and touchdown...\n');

[flight_log, m] = simulatePID(P, wind_cfg);

%% 3. Display Quantitative Performance Metrics
fprintf('\n------------------------------------------------------------------------\n');
fprintf('  DISTURBANCE REJECTION & TOUCHDOWN PERFORMANCE\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('  Landing Status:          %s\n', m.status);
fprintf('  Mission Success:         %s\n', mat2str(m.is_success));
fprintf('  Touchdown Position (x):  %.2f m (Aim: %.1f m, Error: %+.2f m)\n', ...
        m.touchdown_x, P.touchdown_aim_x, m.touchdown_pos_error);
fprintf('  Touchdown Sink Rate:     %.2f m/s (Soft: <= %.1f m/s, Max Safe: <= %.1f m/s)\n', ...
        m.touchdown_sink, P.td_sink_rate_soft, P.td_sink_rate_max);
fprintf('  Touchdown Airspeed:      %.2f m/s (Target: %.1f m/s, Error: %+.2f m/s)\n', ...
        m.touchdown_V, P.V_td, m.touchdown_speed_error);
fprintf('  Touchdown Pitch Angle:   %.2f deg (Target: %.1f deg, Limits: [%.1f, %.1f] deg)\n', ...
        rad2deg(m.touchdown_pitch), rad2deg(P.theta_flare_tgt), ...
        rad2deg(P.td_theta_min), rad2deg(P.td_theta_max));
fprintf('  RMS Glide-Slope Error:   %.3f m\n', m.rms_glideslope_error);
fprintf('  Max Glide-Slope Error:   %.3f m\n', m.max_glideslope_error);
fprintf('  RMS Airspeed Error:      %.3f m/s\n', m.rms_airspeed_error);
fprintf('  Total Control Effort:    %.2f\n', m.total_control_effort);
fprintf('  Elevator Smoothness:     %.2f\n', m.elevator_smoothness);
fprintf('  Throttle Smoothness:     %.2f\n', m.throttle_smoothness);
fprintf('  Time to Touchdown:       %.2f s\n', m.time_to_touchdown);
fprintf('------------------------------------------------------------------------\n');

%% 4. Generate & Save Figures
fprintf('Generating publication-quality figures...\n');
plotResults('trajectory', flight_log, P, ...
    sprintf('PID Landing Trajectory under %s', desc_str), 'pid_wind_trajectory.png');
plotResults('time_history', flight_log, P, ...
    sprintf('PID Flight Telemetry under %s', desc_str), 'pid_wind_telemetry.png');

%% 5. Save Results Data
res_file = fullfile(data_dir, 'pid_wind_results.mat');
save(res_file, 'flight_log', 'm', 'P', 'wind_cfg');
fprintf('Saved wind simulation telemetry and metrics to:\n  %s\n', res_file);
fprintf('========================================================================\n\n');

if nargout >= 1, varargout{1} = flight_log; end
if nargout >= 2, varargout{2} = m; end

end
