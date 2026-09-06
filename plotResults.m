function varargout = plotResults(cmd, varargin)
% PLOTRESULTS Publication-quality plotting suite for aircraft landing simulations.
%
% Generates publication-grade figures with standardized fonts, LaTeX/TeX
% annotations, distinct colors, and exports high-resolution PNG/PDF assets.
%
% Subcommands:
%   plotResults('trajectory', flight_log, P, title_str, save_name)
%   plotResults('time_history', flight_log, P, title_str, save_name)
%   plotResults('comparison', pid_log, rl_log, P, save_name)
%   plotResults('monte_carlo', mc_summary, save_name)

fig_dir = fullfile(pwd, 'Results', 'Figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

switch lower(cmd)
    case 'trajectory'
        h_fig = plotTrajectory(fig_dir, varargin{:});
        if nargout > 0, varargout{1} = h_fig; end
    case 'time_history'
        h_fig = plotTimeHistory(fig_dir, varargin{:});
        if nargout > 0, varargout{1} = h_fig; end
    case 'comparison'
        h_fig = plotComparison(fig_dir, varargin{:});
        if nargout > 0, varargout{1} = h_fig; end
    case 'monte_carlo'
        h_fig = plotMonteCarlo(fig_dir, varargin{:});
        if nargout > 0, varargout{1} = h_fig; end
    otherwise
        error('plotResults:UnknownCommand', 'Unknown subcommand: %s', cmd);
end

end

%% 1. Approach & Landing Trajectory Plot (h vs x)
function fig = plotTrajectory(fig_dir, flight_log, P, title_str, save_name)
if nargin < 3 || isempty(P), P = aircraftParameters(); end
if nargin < 4 || isempty(title_str), title_str = 'Aircraft Landing Trajectory Profile'; end
if nargin < 5 || isempty(save_name), save_name = 'trajectory_profile.png'; end

fig = figure('Color', 'w', 'Position', [100, 100, 1000, 520], 'Visible', 'on');
hold on; grid on; box on;

% Runway Ground and Touchdown Aim Point
plot([-500, 1500], [0, 0], 'Color', [0.3 0.3 0.3], 'LineWidth', 3, 'DisplayName', 'Runway Surface');
plot([0, 0], [-5, 20], 'Color', [0.2 0.7 0.2], 'LineWidth', 2.5, 'DisplayName', 'Runway Threshold (x=0)');
plot(P.touchdown_aim_x, 0, 'kp', 'MarkerSize', 14, 'MarkerFaceColor', [1 0.8 0.1], ...
     'LineWidth', 1.5, 'DisplayName', 'Aim Point (x=300m)');

% Approach Corridor & Reference Glide-Slope
x_grid = linspace(-3200, 500, 300);
corridor = landingScenario('corridor', x_grid, P);
fill([x_grid, fliplr(x_grid)], [corridor.h_up, fliplr(corridor.h_low)], ...
     [0.85 0.92 1.0], 'FaceAlpha', 0.45, 'EdgeColor', 'none', 'DisplayName', 'Approach Corridor');

plot(flight_log.x, flight_log.h_ref, 'k--', 'LineWidth', 1.8, 'DisplayName', 'Reference Profile (3° GS + Flare)');

% Actual Aircraft Flight Path
plot(flight_log.x, flight_log.h, 'b-', 'LineWidth', 2.2, 'DisplayName', 'Aircraft Trajectory');

% Flare Initiation Marker and Touchdown Point
idx_flare = find(flight_log.h <= P.h_flare, 1);
if ~isempty(idx_flare)
    plot(flight_log.x(idx_flare), flight_log.h(idx_flare), 'mo', 'MarkerSize', 10, ...
         'MarkerFaceColor', 'm', 'DisplayName', sprintf('Flare Point (h=%.0fm)', P.h_flare));
end

plot(flight_log.x(end), flight_log.h(end), 'rs', 'MarkerSize', 12, ...
     'MarkerFaceColor', 'r', 'DisplayName', sprintf('Touchdown (x=%.1fm, hdot=%.2fm/s)', ...
     flight_log.x(end), flight_log.sink_rate(end)));

xlabel('Longitudinal Distance x [m]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Altitude h [m]', 'FontSize', 12, 'FontWeight', 'bold');
title(title_str, 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
xlim([-3100, 800]);
ylim([-5, 200]);

savePlot(fig, fullfile(fig_dir, save_name));
end

%% 2. Detailed Multi-Panel Time Histories
function fig = plotTimeHistory(fig_dir, flight_log, P, title_str, save_name)
if nargin < 3 || isempty(P), P = aircraftParameters(); end
if nargin < 4 || isempty(title_str), title_str = 'Flight Telemetry Time Histories'; end
if nargin < 5 || isempty(save_name), save_name = 'flight_time_history.png'; end

t = flight_log.time;
fig = figure('Color', 'w', 'Position', [80, 50, 1100, 750], 'Visible', 'on');

% 1. Altitude & Reference
subplot(3, 2, 1); hold on; grid on; box on;
plot(t, flight_log.h_ref, 'k--', 'LineWidth', 1.5, 'DisplayName', 'h_{ref}');
plot(t, flight_log.h, 'b-', 'LineWidth', 1.8, 'DisplayName', 'Altitude h');
yline(P.h_flare, 'm:', 'Flare Alt', 'LineWidth', 1.2);
ylabel('Altitude [m]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
title('Altitude Tracking', 'FontSize', 11, 'FontWeight', 'bold');

% 2. Airspeed Tracking
subplot(3, 2, 2); hold on; grid on; box on;
plot(t, flight_log.V_ref, 'k--', 'LineWidth', 1.5, 'DisplayName', 'V_{ref}');
plot(t, flight_log.V, 'Color', [0 0.5 0], 'LineWidth', 1.8, 'DisplayName', 'Airspeed V');
yline(P.V_stall, 'r:', 'V_{stall}', 'LineWidth', 1.2);
ylabel('Airspeed [m/s]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
title('Airspeed Regulation', 'FontSize', 11, 'FontWeight', 'bold');

% 3. Pitch Attitude and Flight-Path Angle
subplot(3, 2, 3); hold on; grid on; box on;
plot(t, rad2deg(flight_log.theta), 'b-', 'LineWidth', 1.8, 'DisplayName', 'Pitch \theta');
plot(t, rad2deg(flight_log.gamma), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Flight-Path \gamma');
plot(t, rad2deg(flight_log.alpha), 'k-.', 'LineWidth', 1.2, 'DisplayName', 'AoA \alpha');
yline(rad2deg(P.alpha_stall_pos), 'r:', 'Stall \alpha', 'LineWidth', 1.1);
ylabel('Angles [deg]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
title('Attitude & Aerodynamic Angles', 'FontSize', 11, 'FontWeight', 'bold');

% 4. Vertical Sink Rate
subplot(3, 2, 4); hold on; grid on; box on;
plot(t, flight_log.hdot_ref, 'k--', 'LineWidth', 1.5, 'DisplayName', 'hdot_{ref}');
plot(t, -flight_log.sink_rate, 'm-', 'LineWidth', 1.8, 'DisplayName', 'dh/dt');
yline(-P.td_sink_rate_soft, 'g:', 'Soft TD Limit', 'LineWidth', 1.2);
yline(-P.td_sink_rate_max, 'r:', 'Max Safe TD', 'LineWidth', 1.2);
ylabel('Vertical Speed [m/s]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
title('Vertical Sink Rate', 'FontSize', 11, 'FontWeight', 'bold');

% 5. Elevator Deflection
subplot(3, 2, 5); hold on; grid on; box on;
plot(t, rad2deg(flight_log.delta_e), 'b-', 'LineWidth', 1.6);
yline(rad2deg(P.elevator_max), 'k:');
yline(rad2deg(P.elevator_min), 'k:');
xlabel('Time [s]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Elevator \delta_e [deg]', 'FontSize', 10, 'FontWeight', 'bold');
title('Elevator Deflection', 'FontSize', 11, 'FontWeight', 'bold');

% 6. Throttle Fraction
subplot(3, 2, 6); hold on; grid on; box on;
plot(t, flight_log.delta_T * 100, 'Color', [0.8 0.4 0], 'LineWidth', 1.6);
ylim([0, 105]);
xlabel('Time [s]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Throttle \delta_T [%]', 'FontSize', 10, 'FontWeight', 'bold');
title('Throttle Setting', 'FontSize', 11, 'FontWeight', 'bold');

sgtitle(title_str, 'FontSize', 13, 'FontWeight', 'bold');
savePlot(fig, fullfile(fig_dir, save_name));
end

%% 3. Classical PID vs Deep RL Comparative Overlay
function fig = plotComparison(fig_dir, pid_log, rl_log, P, save_name)
if nargin < 4 || isempty(P), P = aircraftParameters(); end
if nargin < 5 || isempty(save_name), save_name = 'pid_vs_rl_comparison.png'; end

fig = figure('Color', 'w', 'Position', [100, 80, 1100, 700], 'Visible', 'on');

% 1. Trajectory Overlay
subplot(2, 2, 1); hold on; grid on; box on;
plot([-500, 1200], [0, 0], 'k-', 'LineWidth', 2.5, 'DisplayName', 'Runway');
plot(pid_log.x, pid_log.h, 'b-', 'LineWidth', 2.0, 'DisplayName', 'PID Baseline');
plot(rl_log.x, rl_log.h, 'r-', 'LineWidth', 2.0, 'DisplayName', 'RL Policy');
plot(pid_log.x, pid_log.h_ref, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Reference');
xlabel('Distance x [m]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Altitude h [m]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
title('Landing Trajectories Comparison', 'FontSize', 11, 'FontWeight', 'bold');
xlim([-3100, 600]);

% 2. Glide-Slope Tracking Error
subplot(2, 2, 2); hold on; grid on; box on;
eh_pid = pid_log.h_ref - pid_log.h;
eh_rl  = rl_log.h_ref - rl_log.h;
plot(pid_log.time, eh_pid, 'b-', 'LineWidth', 1.8, 'DisplayName', 'PID e_h');
plot(rl_log.time, eh_rl, 'r-', 'LineWidth', 1.8, 'DisplayName', 'RL e_h');
yline(0, 'k:');
xlabel('Time [s]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Altitude Error e_h [m]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
title('Glide-Slope Tracking Error', 'FontSize', 11, 'FontWeight', 'bold');

% 3. Airspeed Tracking Comparison
subplot(2, 2, 3); hold on; grid on; box on;
plot(pid_log.time, pid_log.V, 'b-', 'LineWidth', 1.8, 'DisplayName', 'PID Airspeed');
plot(rl_log.time, rl_log.V, 'r-', 'LineWidth', 1.8, 'DisplayName', 'RL Airspeed');
plot(pid_log.time, pid_log.V_ref, 'k--', 'LineWidth', 1.2, 'DisplayName', 'V_{ref}');
xlabel('Time [s]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Airspeed [m/s]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
title('Airspeed Response', 'FontSize', 11, 'FontWeight', 'bold');

% 4. Control Action Comparison (Elevator)
subplot(2, 2, 4); hold on; grid on; box on;
plot(pid_log.time, rad2deg(pid_log.delta_e), 'b-', 'LineWidth', 1.6, 'DisplayName', 'PID \delta_e');
plot(rl_log.time, rad2deg(rl_log.delta_e), 'r-', 'LineWidth', 1.6, 'DisplayName', 'RL \delta_e');
xlabel('Time [s]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Elevator \delta_e [deg]', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
title('Control Action Comparison', 'FontSize', 11, 'FontWeight', 'bold');

sgtitle('Quantitative Comparison: Classical PID Baseline vs Reinforcement Learning', ...
        'FontSize', 13, 'FontWeight', 'bold');
savePlot(fig, fullfile(fig_dir, save_name));
end

%% 4. Monte Carlo Statistical Summary Plot
function fig = plotMonteCarlo(fig_dir, mc_summary, save_name)
if nargin < 3 || isempty(save_name), save_name = 'monte_carlo_dispersion.png'; end

num_suites = length(mc_summary);
fig = figure('Color', 'w', 'Position', [100, 100, 1050, 650], 'Visible', 'on');

% 1. Success Rate Comparison Bar Chart
subplot(2, 2, 1); hold on; grid on; box on;
names = {mc_summary.name};
success_rates = [mc_summary.success_rate];
b = bar(categorical(names), success_rates, 0.6);
b.FaceColor = [0.15 0.55 0.85];
ylabel('Success Rate [%]', 'FontSize', 10, 'FontWeight', 'bold');
ylim([0, 105]);
title('Landing Mission Success Rate', 'FontSize', 11, 'FontWeight', 'bold');
xtickangle(25);

% 2. Touchdown Position Error (Box / Errorbar)
subplot(2, 2, 2); hold on; grid on; box on;
pos_means = [mc_summary.mean_pos_err];
pos_stds  = [mc_summary.std_pos_err];
errorbar(1:num_suites, pos_means, pos_stds, 'o-', 'LineWidth', 1.8, ...
         'MarkerSize', 8, 'MarkerFaceColor', 'b', 'Color', 'b');
xticks(1:num_suites);
xticklabels(names);
xtickangle(25);
ylabel('Position Error \Delta x_{td} [m]', 'FontSize', 10, 'FontWeight', 'bold');
yline(0, 'k:');
title('Touchdown Longitudinal Dispersion', 'FontSize', 11, 'FontWeight', 'bold');

% 3. Touchdown Sink Rate (Mean +/- Std)
subplot(2, 2, 3); hold on; grid on; box on;
sink_means = [mc_summary.mean_sink_rate];
sink_stds  = [mc_summary.std_sink_rate];
errorbar(1:num_suites, sink_means, sink_stds, 's-', 'LineWidth', 1.8, ...
         'MarkerSize', 8, 'MarkerFaceColor', 'm', 'Color', 'm');
yline(1.0, 'g:', 'Soft Landing Limit (1.0 m/s)', 'LineWidth', 1.2);
yline(1.8, 'r:', 'Structural Limit (1.8 m/s)', 'LineWidth', 1.2);
xticks(1:num_suites);
xticklabels(names);
xtickangle(25);
ylabel('Touchdown Sink Rate [m/s]', 'FontSize', 10, 'FontWeight', 'bold');
title('Touchdown Vertical Impact Sink Rate', 'FontSize', 11, 'FontWeight', 'bold');

% 4. Total Control Effort Comparison
subplot(2, 2, 4); hold on; grid on; box on;
eff_means = [mc_summary.mean_effort];
eff_stds  = [mc_summary.std_effort];
errorbar(1:num_suites, eff_means, eff_stds, 'd-', 'LineWidth', 1.8, ...
         'MarkerSize', 8, 'MarkerFaceColor', [0.8 0.4 0], 'Color', [0.8 0.4 0]);
xticks(1:num_suites);
xticklabels(names);
xtickangle(25);
ylabel('Total Control Effort \int u^2 dt', 'FontSize', 10, 'FontWeight', 'bold');
title('Actuator Control Effort Consumption', 'FontSize', 11, 'FontWeight', 'bold');

sgtitle('Monte Carlo Robustness & Dispersion Analysis', 'FontSize', 13, 'FontWeight', 'bold');
savePlot(fig, fullfile(fig_dir, save_name));
end

%% Helper: High-Resolution Figure Export
function savePlot(fig, filepath)
try
    saveas(fig, filepath);
    fprintf('[Plot Export] Successfully saved figure to: %s\n', filepath);
catch ME
    fprintf('[Plot Export] Save warning: %s\n', ME.message);
end
end
