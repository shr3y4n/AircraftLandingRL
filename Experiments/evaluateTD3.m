function varargout = evaluateTD3(checkpoint_path)
% EVALUATETD3 Evaluates a trained TD3 reinforcement learning landing policy.
%
% Runs deterministic policy evaluation (zero exploration noise) under both calm air
% and atmospheric wind shear conditions, logging performance metrics and plotting
% trajectory and control time histories.
%
% Inputs:
%   checkpoint_path - (Optional) Path to specific .mat checkpoint file.
%                     If omitted, automatically loads the latest TD3 checkpoint.
%
% Outputs:
%   m_nom      - Performance metrics under nominal calm air
%   m_wind     - Performance metrics under atmospheric wind shear
%   flight_log - Time-series telemetry struct from the nominal flight
%
% Artifacts saved to:
%   Results/Data/td3_evaluation_results.mat
%   Results/Figures/td3_trajectory.png
%   Results/Figures/td3_telemetry.png
%   Results/Figures/td3_wind_trajectory.png

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
fprintf('  EXPERIMENT: TD3 REINFORCEMENT LEARNING POLICY EVALUATION\n');
fprintf('========================================================================\n');

%% 2. Locate and Load Agent Checkpoint
P = aircraftParameters();

if nargin < 1 || isempty(checkpoint_path)
    % Look for final checkpoint first
    target_chk = fullfile(chk_dir, 'agent_td3_final.mat');
    if ~exist(target_chk, 'file')
        % Look for latest episode checkpoint
        chk_files = dir(fullfile(chk_dir, 'agent_td3_ep*.mat'));
        if isempty(chk_files)
            error('evaluateTD3:NoCheckpoint', ...
                  ['No trained TD3 checkpoint found in:\n  %s\n' ...
                   'Please execute trainTD3("train") or trainTD3("test") before evaluating.'], chk_dir);
        else
            % Select latest checkpoint
            [~, idx] = max([chk_files.datenum]);
            target_chk = fullfile(chk_files(idx).folder, chk_files(idx).name);
        end
    end
else
    target_chk = checkpoint_path;
    if ~exist(target_chk, 'file')
        error('evaluateTD3:FileNotFound', 'Specified checkpoint does not exist: %s', target_chk);
    end
end

fprintf('Loading TD3 policy from:\n  %s\n\n', target_chk);
agent = rlEngine('load', target_chk);

%% 3. Nominal Flight Evaluation (Calm Air)
fprintf('Evaluating TD3 policy under calm air (wind_mode = none)...\n');
wind_nom.mode = 'none';
[m_nom, log_nom] = evaluateAgent(agent, P, wind_nom);

fprintf('\n------------------------------------------------------------------------\n');
fprintf('  NOMINAL TD3 LANDING PERFORMANCE\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('  Landing Status:          %s\n', m_nom.status);
fprintf('  Mission Success:         %s\n', mat2str(m_nom.is_success));
fprintf('  Touchdown Position (x):  %.2f m (Aim: %.1f m, Error: %+.2f m)\n', ...
        m_nom.touchdown_x, P.touchdown_aim_x, m_nom.touchdown_pos_error);
fprintf('  Touchdown Sink Rate:     %.2f m/s (Limit <= %.1f m/s)\n', ...
        m_nom.touchdown_sink, P.td_sink_rate_max);
fprintf('  Touchdown Airspeed:      %.2f m/s (Target: %.1f m/s, Error: %+.2f m/s)\n', ...
        m_nom.touchdown_V, P.V_td, m_nom.touchdown_speed_error);
fprintf('  Touchdown Pitch Deck:    %.2f deg\n', rad2deg(m_nom.touchdown_pitch));
fprintf('  RMS Glide-Slope Error:   %.3f m\n', m_nom.rms_glideslope_error);
fprintf('  RMS Airspeed Error:      %.3f m/s\n', m_nom.rms_airspeed_error);
fprintf('  Total Control Effort:    %.2f\n', m_nom.total_control_effort);
fprintf('------------------------------------------------------------------------\n');

%% 4. Disturbance Flight Evaluation (Wind Shear)
fprintf('\nEvaluating TD3 policy under atmospheric wind shear (V_headwind = 6.0 m/s)...\n');
wind_shear.mode = 'shear';
wind_shear.V_headwind = 6.0;
[m_wind, log_wind] = evaluateAgent(agent, P, wind_shear);

fprintf('\n------------------------------------------------------------------------\n');
fprintf('  WIND SHEAR TD3 LANDING PERFORMANCE\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('  Landing Status:          %s\n', m_wind.status);
fprintf('  Mission Success:         %s\n', mat2str(m_wind.is_success));
fprintf('  Touchdown Position (x):  %.2f m (Aim: %.1f m, Error: %+.2f m)\n', ...
        m_wind.touchdown_x, P.touchdown_aim_x, m_wind.touchdown_pos_error);
fprintf('  Touchdown Sink Rate:     %.2f m/s (Limit <= %.1f m/s)\n', ...
        m_wind.touchdown_sink, P.td_sink_rate_max);
fprintf('  RMS Glide-Slope Error:   %.3f m\n', m_wind.rms_glideslope_error);
fprintf('  RMS Airspeed Error:      %.3f m/s\n', m_wind.rms_airspeed_error);
fprintf('------------------------------------------------------------------------\n');

%% 5. Generate & Save Figures
fprintf('\nGenerating publication-quality figures...\n');
plotResults('trajectory', log_nom, P, ...
    'TD3 Neural Policy Landing Trajectory (Nominal)', 'td3_trajectory.png');
plotResults('time_history', log_nom, P, ...
    'TD3 Neural Policy Closed-Loop Flight Telemetry', 'td3_telemetry.png');
plotResults('trajectory', log_wind, P, ...
    'TD3 Landing Trajectory under Wind Shear', 'td3_wind_trajectory.png');

%% 6. Save Structured Evaluation Results
eval_file = fullfile(data_dir, 'td3_evaluation_results.mat');
save(eval_file, 'm_nom', 'm_wind', 'log_nom', 'log_wind', 'P');
fprintf('Saved evaluation results to:\n  %s\n', eval_file);
fprintf('========================================================================\n\n');

if nargout >= 1, varargout{1} = m_nom; end
if nargout >= 2, varargout{2} = m_wind; end
if nargout >= 3, varargout{3} = log_nom; end

end
