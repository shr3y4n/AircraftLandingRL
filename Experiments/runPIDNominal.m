function varargout = runPIDNominal()
% RUNPIDNOMINAL Executes classical PID baseline approach and landing under calm air.
%
% Simulates the multi-loop flight control system (pitch attitude, glide-slope tracking,
% flare guidance, and autothrottle) in zero-wind conditions.
%
% Outputs:
%   flight_log - Time-series flight telemetry struct
%   m          - Quantitative landing performance metrics struct from metrics.m
%
% Outputs are saved to:
%   Results/Data/pid_nominal_results.mat
%   Results/Figures/pid_nominal_trajectory.png
%   Results/Figures/pid_nominal_telemetry.png

%% 1. Environment & Path Setup
exp_dir   = fileparts(mfilename('fullpath'));
repo_root = fileparts(exp_dir);
addpath(genpath(repo_root));

data_dir = fullfile(repo_root, 'Results', 'Data');
fig_dir  = fullfile(repo_root, 'Results', 'Figures');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(fig_dir, 'dir'),  mkdir(fig_dir);  end

fprintf('========================================================================\n');
fprintf('  EXPERIMENT: CLASSICAL PID BASELINE (NOMINAL / CALM AIR)\n');
fprintf('========================================================================\n');

%% 2. Simulation Execution
P = aircraftParameters();

wind_cfg = struct();
wind_cfg.mode = 'none';

fprintf('Simulating closed-loop approach, flare, and touchdown...\n');
[flight_log, m] = simulatePID(P, wind_cfg);

%% 3. Display Quantitative Performance Metrics
fprintf('\n------------------------------------------------------------------------\n');
fprintf('  TOUCHDOWN & TRACKING PERFORMANCE SUMMARY\n');
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
    'Nominal Classical PID Landing Trajectory', 'pid_nominal_trajectory.png');
plotResults('time_history', flight_log, P, ...
    'Nominal Classical PID Closed-Loop Flight Telemetry', 'pid_nominal_telemetry.png');

%% 5. Save Results Data
res_file = fullfile(data_dir, 'pid_nominal_results.mat');
save(res_file, 'flight_log', 'm', 'P');
fprintf('Saved nominal simulation telemetry and metrics to:\n  %s\n', res_file);
fprintf('========================================================================\n\n');

if nargout >= 1, varargout{1} = flight_log; end
if nargout >= 2, varargout{2} = m; end

end
