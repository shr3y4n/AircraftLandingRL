function [agent, train_stats] = trainTD3(mode, num_episodes)
% TRAINTD3 Trains a Twin Delayed Deep Deterministic Policy Gradient (TD3) landing agent.
%
% Integrates domain randomization across atmospheric wind shear, gusts, and initial
% approach states to produce a robust, safe autonomous landing policy.
%
% Usage:
%   trainTD3()                 % Runs full training (configured episodes from rlConfig)
%   trainTD3("test")           % Runs 3-episode quick smoke test
%   trainTD3("train", 200)     % Runs training for 200 episodes
%   [agent, stats] = trainTD3(...)
%
% Outputs:
%   agent       - Trained TD3 agent struct (standalone pure-MATLAB rlEngine)
%   train_stats - Training history struct with rewards, success rates, and losses
%
% Artifacts saved to:
%   Results/Data/checkpoints/agent_td3_final.mat
%   Results/Data/training_stats_td3.mat
%   Results/Figures/td3_training_curves.png

if nargin < 1 || isempty(mode),         mode = "train"; end
if nargin < 2 || isempty(num_episodes), num_episodes = []; end

%% 1. Path & Directory Setup
exp_dir   = fileparts(mfilename('fullpath'));
repo_root = fileparts(exp_dir);
addpath(genpath(repo_root));

data_dir = fullfile(repo_root, 'Results', 'Data');
chk_dir  = fullfile(data_dir, 'checkpoints');
fig_dir  = fullfile(repo_root, 'Results', 'Figures');
if ~exist(chk_dir, 'dir'), mkdir(chk_dir); end
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

fprintf('========================================================================\n');
fprintf('  EXPERIMENT: TD3 REINFORCEMENT LEARNING TRAINING (%s MODE)\n', upper(string(mode)));
fprintf('========================================================================\n');

%% 2. Configure Training Parameters
P   = aircraftParameters();
cfg = rlConfig();

if ~isempty(num_episodes)
    if lower(string(mode)) == "test"
        cfg.test_episodes = num_episodes;
    else
        cfg.max_episodes = num_episodes;
    end
end

% Ensure reproducible seed
rng(100, 'twister');

%% 3. Execute Training Pipeline
[agent, train_stats] = trainAgent(mode, "td3", P, cfg);

%% 4. Save Final Agent Checkpoint
final_chk = fullfile(chk_dir, 'agent_td3_final.mat');
if isstruct(agent) && isfield(agent, 'actor')
    rlEngine('save', agent, final_chk);
else
    save(final_chk, 'agent');
end
fprintf('[TD3 Experiment] Final model checkpoint saved to:\n  %s\n', final_chk);

%% 5. Plot and Save Learning Curves
plotTrainingCurves(train_stats, fig_dir, 'td3_training_curves.png');

fprintf('========================================================================\n\n');

end

%% Local Function: Plot Learning Curves
function plotTrainingCurves(stats, fig_dir, save_name)
episodes = 1:length(stats.episode_rewards);
if length(episodes) < 2
    return;
end

fig = figure('Color', 'w', 'Position', [100, 100, 1000, 700], 'Visible', 'on');

% 1. Episode Rewards & Trend
subplot(2, 2, 1); hold on; grid on; box on;
plot(episodes, stats.episode_rewards, 'Color', [0.7 0.85 1.0], 'LineWidth', 1.0, 'DisplayName', 'Raw Reward');
win = max(3, min(20, round(length(episodes) * 0.1)));
if length(stats.episode_rewards) >= win
    smooth_rew = movmean(stats.episode_rewards, win);
    plot(episodes, smooth_rew, 'b-', 'LineWidth', 2.0, 'DisplayName', sprintf('Moving Avg (%d eps)', win));
end
xlabel('Episode', 'FontWeight', 'bold');
ylabel('Episode Return', 'FontWeight', 'bold');
title('Training Reward Progression', 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 9);

% 2. Success Rate (Cumulative & Windowed)
subplot(2, 2, 2); hold on; grid on; box on;
cum_success = cumsum(stats.success_history) ./ episodes(:) * 100.0;
plot(episodes, cum_success, 'm-', 'LineWidth', 2.0, 'DisplayName', 'Cumulative Success Rate');
if length(stats.success_history) >= win
    win_success = movmean(stats.success_history * 100.0, win);
    plot(episodes, win_success, 'Color', [0 0.6 0.2], 'LineWidth', 1.8, 'DisplayName', sprintf('Rolling Window (%d eps)', win));
end
ylim([0, 105]);
xlabel('Episode', 'FontWeight', 'bold');
ylabel('Success Rate [%]', 'FontWeight', 'bold');
title('Landing Mission Success Rate', 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 9);

% 3. Episode Steps (Survival Duration)
subplot(2, 2, 3); hold on; grid on; box on;
plot(episodes, stats.episode_lengths, 'Color', [0.8 0.4 0], 'LineWidth', 1.5);
xlabel('Episode', 'FontWeight', 'bold');
ylabel('Steps per Episode', 'FontWeight', 'bold');
title('Episode Survival Duration', 'FontWeight', 'bold');

% 4. Critic Loss (if recorded)
subplot(2, 2, 4); hold on; grid on; box on;
if isfield(stats, 'critic_loss') && any(stats.critic_loss > 0)
    plot(episodes, stats.critic_loss, 'r-', 'LineWidth', 1.5);
    ylabel('Mean Squared Bellman Error', 'FontWeight', 'bold');
else
    text(0.5, 0.5, 'Critic Loss Tracked in Mini-Batches', 'HorizontalAlignment', 'center');
end
xlabel('Episode', 'FontWeight', 'bold');
title('Critic Loss Convergence', 'FontWeight', 'bold');

sgtitle('TD3 Autonomous Aircraft Landing: Training Convergence', 'FontSize', 13, 'FontWeight', 'bold');

try
    saveas(fig, fullfile(fig_dir, save_name));
    fprintf('[TD3 Experiment] Saved training curve plot to:\n  %s\n', fullfile(fig_dir, save_name));
catch ME
    fprintf('[TD3 Experiment] Warning saving plot: %s\n', ME.message);
end
end
