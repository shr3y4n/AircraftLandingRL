function comp_table = compareControllers(wind_mode)
% COMPARECONTROLLERS Head-to-head benchmarking of PID, TD3, and SAC controllers.
%
% Dynamically inspects available checkpoints, executes simultaneous flight evaluations
% under identical disturbance conditions, overlays trajectories & actuator telemetry,
% and exports structured comparison data to CSV and publication LaTeX tables.
%
% Inputs:
%   wind_mode - 'none' (calm air), 'shear' (default), 'gust', or 'turbulence'
%
% Outputs:
%   comp_table - MATLAB table containing quantitative performance metrics across controllers
%
% Artifacts saved to:
%   Results/Data/controller_comparison.csv
%   Results/Data/controller_comparison.tex
%   Results/Data/controller_comparison.mat
%   Results/Figures/controller_comparison_overlay.png

if nargin < 1 || isempty(wind_mode), wind_mode = 'shear'; end

%% 1. Path & Directory Setup
exp_dir   = fileparts(mfilename('fullpath'));
repo_root = fileparts(exp_dir);
addpath(genpath(repo_root));

data_dir = fullfile(repo_root, 'Results', 'Data');
chk_dir  = fullfile(data_dir, 'checkpoints');
fig_dir  = fullfile(repo_root, 'Results', 'Figures');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(fig_dir, 'dir'),  mkdir(fig_dir);  end

fprintf('========================================================================\n');
fprintf('  EXPERIMENT: CONTROLLER BENCHMARKING & COMPARISON (WIND: %s)\n', upper(wind_mode));
fprintf('========================================================================\n');

P = aircraftParameters();

%% 2. Setup Disturbance Configuration
wind_cfg = struct();
switch lower(wind_mode)
    case 'none'
        wind_cfg.mode = 'none';
        wind_desc = 'Calm Air';
    case 'shear'
        wind_cfg.mode = 'shear';
        wind_cfg.V_headwind = 6.0;
        wind_desc = 'Wind Shear (6 m/s Headwind)';
    case 'gust'
        wind_cfg.mode = 'gust';
        wind_cfg.V_headwind = 4.0;
        wind_cfg.gust_Ax    = 4.0;
        wind_cfg.gust_Ah    = -2.0;
        wind_cfg.gust_start = 20.0;
        wind_cfg.gust_dur   = 5.0;
        wind_desc = 'Discrete Gusts';
    case 'turbulence'
        wind_cfg.mode = 'turbulence';
        wind_cfg.V_headwind = 5.0;
        wind_cfg.turb_level = 'moderate';
        wind_cfg.seed = 42;
        wind_desc = 'Dryden Turbulence (Moderate)';
    otherwise
        wind_cfg.mode = 'none';
        wind_desc = 'Calm Air';
end

fprintf('Disturbance Condition: %s\n\n', wind_desc);

%% 3. Evaluate Available Controllers
controllers = {};
logs        = {};
metric_list = {};

% A. Classical PID Baseline (Always Available)
fprintf('--> [1] Simulating Classical PID Baseline...\n');
[pid_log, pid_m] = simulatePID(P, wind_cfg);
controllers{end+1} = 'PID';
logs{end+1}        = pid_log;
metric_list{end+1} = pid_m;
fprintf('    PID Touchdown: x=%.1fm, hdot=%.2fm/s, V=%.1fm/s, Status=%s\n', ...
        pid_m.touchdown_x, pid_m.touchdown_sink, pid_m.touchdown_V, pid_m.status);

% B. TD3 Policy (if trained)
td3_chk = fullfile(chk_dir, 'agent_td3_final.mat');
if ~exist(td3_chk, 'file')
    chk_files = dir(fullfile(chk_dir, 'agent_td3_ep*.mat'));
    if ~isempty(chk_files)
        [~, idx] = max([chk_files.datenum]);
        td3_chk = fullfile(chk_files(idx).folder, chk_files(idx).name);
    end
end

if exist(td3_chk, 'file')
    fprintf('--> [2] Evaluating TD3 Agent (%s)...\n', td3_chk);
    try
        agent_td3 = rlEngine('load', td3_chk);
        [td3_m, td3_log] = evaluateAgent(agent_td3, P, wind_cfg);
        controllers{end+1} = 'TD3';
        logs{end+1}        = td3_log;
        metric_list{end+1} = td3_m;
        fprintf('    TD3 Touchdown: x=%.1fm, hdot=%.2fm/s, V=%.1fm/s, Status=%s\n', ...
                td3_m.touchdown_x, td3_m.touchdown_sink, td3_m.touchdown_V, td3_m.status);
    catch ME
        fprintf('    Notice: Could not evaluate TD3: %s\n', ME.message);
    end
else
    fprintf('--> [2] TD3: Not evaluated yet (no checkpoint in Results/Data/checkpoints/).\n');
end

% C. SAC Policy (if trained)
sac_chk = fullfile(chk_dir, 'agent_sac_final.mat');
if ~exist(sac_chk, 'file')
    chk_files = dir(fullfile(chk_dir, 'agent_sac_ep*.mat'));
    if ~isempty(chk_files)
        [~, idx] = max([chk_files.datenum]);
        sac_chk = fullfile(chk_files(idx).folder, chk_files(idx).name);
    end
end

if exist(sac_chk, 'file')
    fprintf('--> [3] Evaluating SAC Agent (%s)...\n', sac_chk);
    try
        agent_sac = rlEngine('load', sac_chk);
        [sac_m, sac_log] = evaluateAgent(agent_sac, P, wind_cfg);
        controllers{end+1} = 'SAC';
        logs{end+1}        = sac_log;
        metric_list{end+1} = sac_m;
        fprintf('    SAC Touchdown: x=%.1fm, hdot=%.2fm/s, V=%.1fm/s, Status=%s\n', ...
                sac_m.touchdown_x, sac_m.touchdown_sink, sac_m.touchdown_V, sac_m.status);
    catch ME
        fprintf('    Notice: Could not evaluate SAC: %s\n', ME.message);
    end
else
    fprintf('--> [3] SAC: Not evaluated yet (no checkpoint in Results/Data/checkpoints/).\n');
end

%% 4. Compile Quantitative Comparison Table
num_c = length(controllers);
Controller         = string(controllers');
Status             = strings(num_c, 1);
Success            = false(num_c, 1);
Touchdown_x_m      = zeros(num_c, 1);
Delta_x_m          = zeros(num_c, 1);
Touchdown_sink_mps = zeros(num_c, 1);
Touchdown_speed_mps= zeros(num_c, 1);
Touchdown_pitch_deg= zeros(num_c, 1);
RMS_GlideError_m   = zeros(num_c, 1);
RMS_Airspeed_mps   = zeros(num_c, 1);
Control_Effort     = zeros(num_c, 1);
Control_Smoothness = zeros(num_c, 1);

for i = 1:num_c
    m = metric_list{i};
    Status(i)              = string(m.status);
    Success(i)             = m.is_success;
    Touchdown_x_m(i)       = m.touchdown_x;
    Delta_x_m(i)           = m.touchdown_pos_error;
    Touchdown_sink_mps(i)  = m.touchdown_sink;
    Touchdown_speed_mps(i) = m.touchdown_V;
    Touchdown_pitch_deg(i) = rad2deg(m.touchdown_pitch);
    RMS_GlideError_m(i)    = m.rms_glideslope_error;
    RMS_Airspeed_mps(i)    = m.rms_airspeed_error;
    Control_Effort(i)      = m.total_control_effort;
    Control_Smoothness(i)  = m.total_control_smoothness;
end

comp_table = table(Controller, Status, Success, Touchdown_x_m, Delta_x_m, ...
                   Touchdown_sink_mps, Touchdown_speed_mps, Touchdown_pitch_deg, ...
                   RMS_GlideError_m, RMS_Airspeed_mps, Control_Effort, Control_Smoothness);

fprintf('\n------------------------------------------------------------------------\n');
fprintf('  CONTROLLER PERFORMANCE COMPARISON MATRIX (%s)\n', wind_desc);
fprintf('------------------------------------------------------------------------\n');
disp(comp_table);

%% 5. Export to CSV and LaTeX
csv_file = fullfile(data_dir, 'controller_comparison.csv');
writetable(comp_table, csv_file);
fprintf('Exported comparison CSV to:\n  %s\n', csv_file);

tex_file = fullfile(data_dir, 'controller_comparison.tex');
fid = fopen(tex_file, 'w');
if fid > 0
    fprintf(fid, '%% Auto-generated comparative analysis table\n');
    fprintf(fid, '\\begin{table}[htbp]\n\\centering\n');
    fprintf(fid, '\\caption{Controller Benchmarking Performance under %s}\n', wind_desc);
    fprintf(fid, '\\label{tab:controller_comparison}\n');
    fprintf(fid, '\\begin{tabular}{l c c c c c c c}\n');
    fprintf(fid, '\\hline\\hline\n');
    fprintf(fid, 'Controller & Status & $\\Delta x_{\\text{td}}$ [m] & $\\dot{h}_{\\text{td}}$ [m/s] & $V_{\\text{td}}$ [m/s] & $\\text{RMS}(e_h)$ [m] & Effort \\\\\n');
    fprintf(fid, '\\hline\n');
    for i = 1:num_c
        fprintf(fid, '%-10s & %-16s & $%+6.1f$ & $%.2f$ & $%.1f$ & $%.2f$ & $%.1f$ \\\\\n', ...
                Controller(i), Status(i), Delta_x_m(i), Touchdown_sink_mps(i), ...
                Touchdown_speed_mps(i), RMS_GlideError_m(i), Control_Effort(i));
    end
    fprintf(fid, '\\hline\\hline\n\\end{tabular}\n\\end{table}\n');
    fclose(fid);
    fprintf('Exported comparison LaTeX table to:\n  %s\n', tex_file);
end

mat_file = fullfile(data_dir, 'controller_comparison.mat');
save(mat_file, 'comp_table', 'controllers', 'logs', 'metric_list', 'wind_cfg', 'P');

%% 6. Generate Comparative Multi-Panel Overlay Plot
plotComparativeOverlay(controllers, logs, P, fig_dir, 'controller_comparison_overlay.png');

fprintf('========================================================================\n\n');

end

%% Local Plotting Function
function plotComparativeOverlay(names, logs, P, fig_dir, save_name)
colors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.4660 0.6740 0.1880]};
styles = {'-', '--', '-.'};

fig = figure('Color', 'w', 'Position', [80, 50, 1100, 750], 'Visible', 'on');

% 1. Trajectory Profile
subplot(3, 2, 1); hold on; grid on; box on;
plot([-500, 1000], [0, 0], 'Color', [0.4 0.4 0.4], 'LineWidth', 2.5, 'DisplayName', 'Runway');
plot(logs{1}.x, logs{1}.h_ref, 'k:', 'LineWidth', 1.2, 'DisplayName', 'Reference');
for i = 1:length(names)
    c_idx = mod(i-1, length(colors)) + 1;
    plot(logs{i}.x, logs{i}.h, 'Color', colors{c_idx}, 'LineStyle', styles{c_idx}, ...
         'LineWidth', 1.8, 'DisplayName', names{i});
end
xlabel('Distance x [m]', 'FontWeight', 'bold');
ylabel('Altitude h [m]', 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 8);
title('Landing Trajectories', 'FontWeight', 'bold');
xlim([-3100, 600]);

% 2. Glide-Slope Tracking Error
subplot(3, 2, 2); hold on; grid on; box on;
yline(0, 'k:');
for i = 1:length(names)
    c_idx = mod(i-1, length(colors)) + 1;
    eh = logs{i}.h_ref - logs{i}.h;
    plot(logs{i}.time, eh, 'Color', colors{c_idx}, 'LineStyle', styles{c_idx}, ...
         'LineWidth', 1.6, 'DisplayName', names{i});
end
xlabel('Time [s]', 'FontWeight', 'bold');
ylabel('Altitude Error e_h [m]', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
title('Glide-Slope Tracking Error', 'FontWeight', 'bold');

% 3. Airspeed Response
subplot(3, 2, 3); hold on; grid on; box on;
plot(logs{1}.time, logs{1}.V_ref, 'k:', 'LineWidth', 1.2, 'DisplayName', 'V_{ref}');
for i = 1:length(names)
    c_idx = mod(i-1, length(colors)) + 1;
    plot(logs{i}.time, logs{i}.V, 'Color', colors{c_idx}, 'LineStyle', styles{c_idx}, ...
         'LineWidth', 1.6, 'DisplayName', names{i});
end
xlabel('Time [s]', 'FontWeight', 'bold');
ylabel('Airspeed [m/s]', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
title('Airspeed Regulation', 'FontWeight', 'bold');

% 4. Vertical Sink Rate
subplot(3, 2, 4); hold on; grid on; box on;
yline(-P.td_sink_rate_max, 'r:', 'Safe TD Limit', 'LineWidth', 1.2);
for i = 1:length(names)
    c_idx = mod(i-1, length(colors)) + 1;
    plot(logs{i}.time, -logs{i}.sink_rate, 'Color', colors{c_idx}, 'LineStyle', styles{c_idx}, ...
         'LineWidth', 1.6, 'DisplayName', names{i});
end
xlabel('Time [s]', 'FontWeight', 'bold');
ylabel('Vertical Speed [m/s]', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
title('Descent & Flare Sink Rate', 'FontWeight', 'bold');

% 5. Elevator Deflection
subplot(3, 2, 5); hold on; grid on; box on;
yline(rad2deg(P.elevator_max), 'k:');
yline(rad2deg(P.elevator_min), 'k:');
for i = 1:length(names)
    c_idx = mod(i-1, length(colors)) + 1;
    plot(logs{i}.time, rad2deg(logs{i}.delta_e), 'Color', colors{c_idx}, 'LineStyle', styles{c_idx}, ...
         'LineWidth', 1.5, 'DisplayName', names{i});
end
xlabel('Time [s]', 'FontWeight', 'bold');
ylabel('Elevator \delta_e [deg]', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
title('Elevator Control Deflection', 'FontWeight', 'bold');

% 6. Throttle Setting
subplot(3, 2, 6); hold on; grid on; box on;
for i = 1:length(names)
    c_idx = mod(i-1, length(colors)) + 1;
    plot(logs{i}.time, logs{i}.delta_T * 100, 'Color', colors{c_idx}, 'LineStyle', styles{c_idx}, ...
         'LineWidth', 1.5, 'DisplayName', names{i});
end
ylim([0, 105]);
xlabel('Time [s]', 'FontWeight', 'bold');
ylabel('Throttle \delta_T [%]', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
title('Throttle Setting', 'FontWeight', 'bold');

sgtitle('Head-to-Head Flight Control Performance Comparison', 'FontSize', 13, 'FontWeight', 'bold');

try
    saveas(fig, fullfile(fig_dir, save_name));
    fprintf('[Comparison Export] Saved comparative overlay figure to:\n  %s\n', fullfile(fig_dir, save_name));
catch ME
    fprintf('[Comparison Export] Warning saving plot: %s\n', ME.message);
end
end
