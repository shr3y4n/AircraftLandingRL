function cfg = rlConfig()
% RLCONFIG Centralized hyperparameters for Deep Reinforcement Learning agents.
%
% Defines network topologies, optimizer settings, buffer sizes, and training options
% for TD3 (Twin Delayed DDPG) and SAC (Soft Actor-Critic) landing agents.

%% 1. Network Architecture
cfg.obs_dim         = 11;          % Number of continuous observation features
cfg.act_dim         = 2;           % Number of continuous control actions
cfg.hidden_units    = [128, 128];  % Neurons per hidden layer in MLP

%% 2. Optimization and Learning Rates
cfg.actor_lr        = 1e-4;        % Actor network learning rate
cfg.critic_lr       = 3e-4;        % Critic networks learning rate
cfg.l2_reg          = 1e-4;        % L2 weight decay regularization
cfg.gradient_clip   = 1.0;         % Max gradient norm clipping

%% 3. Temporal Difference Learning & Target Updates
cfg.gamma           = 0.99;        % Discount factor
cfg.tau             = 0.005;       % Polyak soft target update coefficient
cfg.policy_delay    = 2;           % TD3 actor update frequency delay (every d critic steps)

%% 4. Experience Replay Buffer
cfg.buffer_capacity = 200000;      % Maximum transitions in circular replay buffer
cfg.batch_size      = 128;         % Mini-batch size for SGD gradient updates
cfg.warmup_steps    = 1000;        % Random exploration steps before gradient updates begin

%% 5. Exploration Noise (TD3)
cfg.noise_sigma     = 0.15;        % Action exploration Gaussian standard deviation
cfg.target_noise    = 0.20;        % Target policy smoothing noise standard deviation
cfg.noise_clip      = 0.40;        % Bound on target smoothing noise

%% 6. SAC Entropy Regularization
cfg.entropy_target  = -cfg.act_dim; % Target entropy for SAC auto-tuning: -dim(A) = -2
cfg.init_alpha      = 0.2;          % Initial temperature parameter alpha
cfg.alpha_lr        = 3e-4;         % Learning rate for temperature alpha

%% 7. Training Execution Settings
cfg.sample_time     = 0.02;        % Agent decision sample time [s] (matches simulation dt)
cfg.max_episodes    = 600;         % Total training episodes in full train mode
cfg.test_episodes   = 3;           % Quick verification smoke-test episodes
cfg.max_steps_ep    = 3500;        % Maximum steps per episode (~70 s flight)
cfg.save_freq       = 25;          % Checkpoint interval (episodes)
cfg.eval_freq       = 10;          % Evaluation interval (episodes)
cfg.eval_episodes   = 5;           % Number of evaluation episodes per benchmark

end
