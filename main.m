%% =========================================================================
% AIRCRAFTRL: AUTONOMOUS AIRCRAFT APPROACH & LANDING FRAMEWORK
% Classical Control (Multi-Loop PID) & Deep Reinforcement Learning (TD3/SAC)
% =========================================================================

clear;
clc;
close all;

% Add project subdirectories to search path
proj_root = fileparts(mfilename('fullpath'));
addpath(proj_root);
addpath(fullfile(proj_root, 'Controllers'));
addpath(fullfile(proj_root, 'RL'));
addpath(fullfile(proj_root, 'Tests'));
addpath(fullfile(proj_root, 'Results'));
addpath(fullfile(proj_root, 'Results', 'Data'));
addpath(fullfile(proj_root, 'Results', 'Figures'));

fprintf('========================================================================\n');
fprintf('  AIRCRAFTRL: AUTONOMOUS AIRCRAFT LANDING SIMULATION & CONTROL\n');
fprintf('========================================================================\n');
fprintf('  1. Basic Open-Loop Simulation & Aerodynamic Trim Verification\n');
fprintf('  2. Classical PID Landing Simulation (Approach + Flare + Touchdown)\n');
fprintf('  3. Deep Reinforcement Learning Training (TD3 Smoke-Test / Train)\n');
fprintf('  4. Evaluate Trained Reinforcement Learning Policy\n');
fprintf('  5. Side-by-Side Benchmark: Classical PID vs Reinforcement Learning\n');
fprintf('  6. Monte Carlo Robustness Campaign (Disturbances & Dispersions)\n');
fprintf('  7. Generate Full Suite of Publication Figures\n');
fprintf('  8. 2D Interactive Flight Animation & Telemetry HUD\n');
fprintf('  9. Run Complete Verification Test Suite\n');
fprintf('========================================================================\n');

% Load Parameters
P = aircraftParameters();

% Interactive selection or default non-interactive execution
default_choice = 2; % Run PID Landing by default
user_choice = default_choice;

% Prompt user if interactive desktop session is active
if feature('ShowFigureWindows')
    try
        choice_str = input(sprintf('Select Option (1-9) [default = %d]: ', default_choice), 's');
        if ~isempty(choice_str)
            user_choice = str2double(choice_str);
        end
    catch
        user_choice = default_choice;
    end
end

fprintf('\n>>> Executing Selected Option: %d <<<\n\n', user_choice);

switch user_choice
    case 1
        %% Option 1: Open-Loop Trim Verification
        fprintf('--- Option 1: Trim & Open-Loop Dynamics ---\n');
        [init_state, trim] = initializeAircraft(P);
        fprintf('Simulating 10 seconds of trimmed glide-slope descent...\n');
        
        sim_time = 10.0;
        dt = P.dt;
        N = round(sim_time / dt);
        s = init_state;
        u = [trim.delta_e; trim.delta_T];
        
        log_h = zeros(N, 1);
        log_V = zeros(N, 1);
        log_t = (0:N-1)' * dt;
        
        for k = 1:N
            log_h(k) = s.h;
            log_V(k) = s.V;
            s = updateAircraft(s, u, dt, P);
        end
        
        figure('Color', 'w');
        subplot(2,1,1); plot(log_t, log_h, 'LineWidth', 1.8); grid on;
        ylabel('Altitude [m]'); title('Open-Loop Trim Descent');
        subplot(2,1,2); plot(log_t, log_V, 'Color', [0 0.5 0], 'LineWidth', 1.8); grid on;
        ylabel('Airspeed [m/s]'); xlabel('Time [s]');
        fprintf('Open-loop simulation complete.\n');

    case 2
        %% Option 2: Classical PID Landing Simulation
        fprintf('--- Option 2: Classical PID Landing Simulation ---\n');
        wind_cfg.mode = 'shear';
        wind_cfg.V_headwind = 6.0;
        
        fprintf('Simulating approach under atmospheric boundary layer wind shear...\n');
        [flight_log, m] = simulatePID(P, wind_cfg);
        
        fprintf('\nLanding Performance Summary:\n');
        fprintf('  Touchdown Status:        %s\n', m.status);
        fprintf('  Touchdown Longitudinal:  x = %.1f m (aim: %.1f m, err: %+.1f m)\n', ...
                m.touchdown_x, P.touchdown_aim_x, m.touchdown_pos_error);
        fprintf('  Touchdown Vertical Sink: hdot = %.2f m/s (safe limit <= %.1f m/s)\n', ...
                m.touchdown_sink, P.td_sink_rate_max);
        fprintf('  Touchdown Speed:         V = %.1f m/s (target: %.1f m/s)\n', ...
                m.touchdown_V, P.V_td);
        fprintf('  Touchdown Pitch Deck:    theta = %.1f deg\n', rad2deg(m.touchdown_pitch));
        fprintf('  RMS Glide-Slope Error:   %.2f m\n', m.rms_glideslope_error);
        fprintf('  Total Control Effort:    %.1f\n', m.total_control_effort);
        
        plotResults('trajectory', flight_log, P, 'PID Approach & Landing Trajectory', 'pid_trajectory.png');
        plotResults('time_history', flight_log, P, 'PID Closed-Loop Flight Telemetry', 'pid_telemetry.png');

    case 3
        %% Option 3: RL Agent Training
        fprintf('--- Option 3: Deep Reinforcement Learning Training ---\n');
        % Run lightweight test mode by default to ensure immediate verification
        train_mode = "test";
        if feature('ShowFigureWindows')
            mode_input = input('Train mode ("test" for 3 eps, "train" for full 600 eps) [test]: ', 's');
            if ~isempty(mode_input), train_mode = string(mode_input); end
        end
        trainAgent(train_mode, "td3", P);

    case 4
        %% Option 4: Evaluate RL Agent
        fprintf('--- Option 4: Evaluate RL Agent ---\n');
        chk_files = dir(fullfile(proj_root, 'Results', 'Data', 'checkpoints', '*.mat'));
        if isempty(chk_files)
            fprintf('No saved checkpoint found. Running 3-episode smoke training first...\n');
            agent = trainAgent("test", "td3", P);
        else
            latest_file = fullfile(chk_files(end).folder, chk_files(end).name);
            agent = rlEngine('load', latest_file);
        end
        
        wind_cfg.mode = 'none';
        [m, flight_log] = evaluateAgent(agent, P, wind_cfg);
        fprintf('\nRL Evaluation Performance:\n');
        fprintf('  Status: %s, Touchdown x = %.1f m, Sink = %.2f m/s\n', ...
                m.status, m.touchdown_x, m.touchdown_sink);
        plotResults('trajectory', flight_log, P, 'RL Landing Trajectory', 'rl_trajectory.png');

    case 5
        %% Option 5: Side-by-Side Comparison: PID vs RL
        fprintf('--- Option 5: Side-by-Side Benchmark: PID vs RL ---\n');
        wind_cfg.mode = 'steady';
        wind_cfg.V_headwind = 5.0;
        
        fprintf('Simulating PID baseline...\n');
        [pid_log, m_pid] = simulatePID(P, wind_cfg);
        
        chk_files = dir(fullfile(proj_root, 'Results', 'Data', 'checkpoints', '*.mat'));
        if isempty(chk_files)
            fprintf('Training lightweight policy for evaluation...\n');
            agent = trainAgent("test", "td3", P);
        else
            agent = rlEngine('load', fullfile(chk_files(end).folder, chk_files(end).name));
        end
        
        fprintf('Simulating RL policy...\n');
        [m_rl, rl_log] = evaluateAgent(agent, P, wind_cfg);
        
        plotResults('comparison', pid_log, rl_log, P, 'pid_vs_rl_comparison.png');
        fprintf('Comparison plot saved to Results/Figures/pid_vs_rl_comparison.png\n');

    case 6
        %% Option 6: Monte Carlo Robustness Campaign
        fprintf('--- Option 6: Monte Carlo Robustness Campaign ---\n');
        trials = 25; % Rapid execution
        mc_summary = runMonteCarlo(trials, [], P);
        plotResults('monte_carlo', mc_summary, 'monte_carlo_dispersion.png');
        exportPaperTables(mc_summary);

    case 7
        %% Option 7: Generate Full Suite of Publication Figures
        fprintf('--- Option 7: Generating Full Publication Figures ---\n');
        wind_cfg.mode = 'shear';
        wind_cfg.V_headwind = 6.0;
        [pid_log, ~] = simulatePID(P, wind_cfg);
        plotResults('trajectory', pid_log, P, 'Longitudinal Landing Approach & Flare', 'figure_trajectory.png');
        plotResults('time_history', pid_log, P, 'Longitudinal Flight Telemetry Time Series', 'figure_telemetry.png');
        fprintf('All figures generated in Results/Figures/\n');

    case 8
        %% Option 8: Interactive Flight Animation
        fprintf('--- Option 8: Flight Animation ---\n');
        wind_cfg.mode = 'steady';
        wind_cfg.V_headwind = 4.0;
        [flight_log, ~] = simulatePID(P, wind_cfg);
        animateFlight(flight_log, P, 3.0, false);

    case 9
        %% Option 9: Complete Verification Test Suite
        fprintf('--- Option 9: Running Complete Verification Test Suite ---\n');
        runAllTests();

    otherwise
        error('Invalid selection %d.', user_choice);
end

fprintf('\nDone.\n');