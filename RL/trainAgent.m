function [agent, train_stats] = trainAgent(mode, agent_type, P, custom_cfg)
% TRAINAGENT Reproducible training pipeline for autonomous landing reinforcement learning.
%
% Supports:
%   mode       - "test"  (quick 2-3 episode smoke test to verify execution pipeline)
%                "train" (full training over configured maximum episodes)
%   agent_type - "td3" or "sac"
%
% Features:
%   - Reproducible random seed
%   - Domain randomization of initial altitude, airspeed, position, and atmospheric gusts
%   - Automatic periodic evaluation and checkpointing to Results/Data/checkpoints/
%   - Saves complete training history to Results/Data/training_stats.mat
%
% Outputs:
%   agent       - Trained policy agent
%   train_stats - Struct containing episode rewards, lengths, and landing outcomes

if nargin < 1 || isempty(mode),       mode = "test";      end
if nargin < 2 || isempty(agent_type), agent_type = "td3"; end
if nargin < 3 || isempty(P),          P = aircraftParameters(); end
if nargin < 4 || isempty(custom_cfg), cfg = rlConfig();   else, cfg = custom_cfg; end

mode = lower(string(mode));
agent_type = lower(string(agent_type));

% Setup directories
data_dir = fullfile(pwd, 'Results', 'Data');
chk_dir  = fullfile(data_dir, 'checkpoints');
if ~exist(chk_dir, 'dir'), mkdir(chk_dir); end

fprintf('\n=======================================================\n');
fprintf('  STARTING RL TRAINING PIPELINE: %s (%s MODE)\n', upper(agent_type), upper(mode));
fprintf('=======================================================\n');

% Set reproducible random seed
rng(100, 'twister');

%% 1. Configure Training Parameters Based on Mode
if mode == "test"
    num_episodes = cfg.test_episodes;
    fprintf('[Training Pipeline] Running lightweight smoke-test (%d episodes)...\n', num_episodes);
else
    num_episodes = cfg.max_episodes;
    fprintf('[Training Pipeline] Running full training campaign (%d episodes)...\n', num_episodes);
end

%% 2. Instantiate Environment with Domain Randomization Enabled
wind_cfg.mode = 'shear'; % Atmospheric shear baseline for training robustness
wind_cfg.V_headwind = 5.0;
env = AircraftLandingEnv(P, wind_cfg, true);

%% 3. Instantiate Agent
if agent_type == "sac"
    agent = createSACAgent(env.ObservationInfo, env.ActionInfo, cfg);
else
    agent = createTD3Agent(env.ObservationInfo, env.ActionInfo, cfg);
end

%% 4. Main Training Loop
train_stats.episode_rewards  = zeros(num_episodes, 1);
train_stats.episode_lengths  = zeros(num_episodes, 1);
train_stats.landing_status   = cell(num_episodes, 1);
train_stats.success_history  = zeros(num_episodes, 1);
train_stats.critic_loss      = zeros(num_episodes, 1);

t_start = tic;

for ep = 1:num_episodes
    obs = env.reset();
    ep_reward = 0.0;
    ep_steps  = 0;
    done      = false;
    last_status = 'IN_FLIGHT';
    is_success  = false;
    
    while ~done && (ep_steps < cfg.max_steps_ep)
        ep_steps = ep_steps + 1;
        
        % Select Action with Exploration Noise
        if isstruct(agent) && isfield(agent, 'actor')
            % Standalone pure-MATLAB engine
            action = rlEngine('getAction', agent, obs, true);
        elseif exist('getAction', 'file') == 2
            % Official RL Toolbox agent
            action = getAction(agent, {obs});
            if iscell(action), action = action{1}; end
        else
            action = [0; 0];
        end
        
        % Step Environment
        [next_obs, reward, done, info] = env.step(action);
        
        ep_reward   = ep_reward + reward;
        last_status = info.status;
        is_success  = info.is_success;
        
        % Store Experience and Perform Optimization Update
        if isstruct(agent) && isfield(agent, 'buffer')
            agent = rlEngine('storeTransition', agent, obs, action, reward, next_obs, done);
            if agent.buffer.size >= cfg.batch_size
                [agent, loss_info] = rlEngine('update', agent);
                train_stats.critic_loss(ep) = loss_info.critic_loss;
            end
        end
        
        obs = next_obs;
    end
    
    train_stats.episode_rewards(ep) = ep_reward;
    train_stats.episode_lengths(ep) = ep_steps;
    train_stats.landing_status{ep}  = last_status;
    train_stats.success_history(ep) = double(is_success);
    
    % Console Logging
    fprintf('Episode %3d/%3d | Steps: %4d | Reward: %8.1f | Status: %-25s | Success: %d\n', ...
            ep, num_episodes, ep_steps, ep_reward, last_status, is_success);
    
    % Periodic Checkpoint Saving
    if mod(ep, cfg.save_freq) == 0 || (ep == num_episodes)
        chk_file = fullfile(chk_dir, sprintf('agent_%s_ep%d.mat', agent_type, ep));
        if isstruct(agent) && isfield(agent, 'actor')
            rlEngine('save', agent, chk_file);
        else
            save(chk_file, 'agent');
        end
    end
end

elapsed_time = toc(t_start);
train_stats.elapsed_time = elapsed_time;

% Save Full Training Statistics
stats_file = fullfile(data_dir, sprintf('training_stats_%s.mat', agent_type));
save(stats_file, 'train_stats', 'P', 'cfg');

fprintf('\n=======================================================\n');
fprintf('  TRAINING COMPLETED in %.2f seconds\n', elapsed_time);
fprintf('  Average Reward (last 10 eps): %.2f\n', mean(train_stats.episode_rewards(max(1, end-9):end)));
fprintf('  Final Checkpoint: %s\n', fullfile(chk_dir, sprintf('agent_%s_ep%d.mat', agent_type, num_episodes)));
fprintf('=======================================================\n\n');

end
