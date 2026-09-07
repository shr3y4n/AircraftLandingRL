function agent = createTD3Agent(obs_info, act_info, cfg)
% CREATETD3AGENT Constructs a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
%
% Integrates actor and dual-critic deep neural network representations.
% Compatible with MATLAB Reinforcement Learning Toolbox rlTD3Agent.
% If the RL Toolbox is not installed, creates a standalone neural policy agent.
%
% Inputs:
%   obs_info - Observation specification from createEnvironment()
%   act_info - Action specification from createEnvironment()
%   cfg      - RL hyperparameters struct from rlConfig()
%
% Outputs:
%   agent - Configured TD3 agent object or standalone policy agent

if nargin < 3 || isempty(cfg)
    cfg = rlConfig();
end

obs_dim = cfg.obs_dim;
act_dim = cfg.act_dim;

%% Check for MATLAB RL Toolbox TD3 Agent Support
has_rl_toolbox = (exist('rlTD3Agent', 'file') == 2 || exist('rlTD3Agent', 'class') == 8);

if has_rl_toolbox
    try
        fprintf('[TD3 Setup] Initializing MATLAB RL Toolbox TD3 networks...\n');
        
        % 1. ACTOR NETWORK ARCHITECTURE
        % Input: Observation [11x1] -> Dense 128 (ReLU) -> Dense 128 (ReLU) -> Dense 2 (Tanh)
        actor_layers = [
            featureInputLayer(obs_dim, 'Normalization', 'none', 'Name', 'obs_in')
            fullyConnectedLayer(cfg.hidden_units(1), 'Name', 'act_fc1')
            reluLayer('Name', 'act_relu1')
            fullyConnectedLayer(cfg.hidden_units(2), 'Name', 'act_fc2')
            reluLayer('Name', 'act_relu2')
            fullyConnectedLayer(act_dim, 'Name', 'act_out')
            tanhLayer('Name', 'act_tanh')
        ];
        actor_net = dlnetwork(layerGraph(actor_layers));
        actor_opts = rlOptimizerOptions('LearnRate', cfg.actor_lr, 'GradientThreshold', cfg.gradient_clip);
        actor = rlContinuousDeterministicActor(actor_net, obs_info, act_info, ...
            'ObservationInputNames', 'obs_in', 'ActionDataNames', 'act_tanh');

        % 2. CRITIC NETWORKS ARCHITECTURE (Dual Twin Critics Q1 & Q2)
        % Q(s, a): Concatenates state and action vectors
        critic_opts = rlOptimizerOptions('LearnRate', cfg.critic_lr, 'GradientThreshold', cfg.gradient_clip);
        
        critic1 = buildCriticNet('q1', obs_dim, act_dim, cfg, obs_info, act_info);
        critic2 = buildCriticNet('q2', obs_dim, act_dim, cfg, obs_info, act_info);

        % 3. TD3 AGENT OPTIONS
        agent_opts = rlTD3AgentOptions();
        agent_opts.SampleTime                 = cfg.sample_time;
        agent_opts.DiscountFactor             = cfg.gamma;
        agent_opts.TargetSmoothFactor         = cfg.tau;
        agent_opts.PolicyUpdateFrequency      = cfg.policy_delay;
        agent_opts.ExperienceBufferLength     = cfg.buffer_capacity;
        agent_opts.MiniBatchSize              = cfg.batch_size;
        agent_opts.ExplorationModel.Variance  = cfg.noise_sigma^2;
        agent_opts.TargetPolicySmoothModel.Variance = cfg.target_noise^2;
        agent_opts.TargetPolicySmoothModel.LowerLimit = -cfg.noise_clip;
        agent_opts.TargetPolicySmoothModel.UpperLimit = cfg.noise_clip;

        agent = rlTD3Agent(actor, [critic1, critic2], agent_opts);
        fprintf('[TD3 Setup] Successfully created official rlTD3Agent.\n');
        return;
    catch ME
        fprintf('[TD3 Setup] RL Toolbox construction notice: %s. Using standalone engine.\n', ME.message);
    end
end

% Fallback: Standalone Pure-MATLAB TD3 Policy Agent
fprintf('[TD3 Setup] Initializing standalone pure-MATLAB TD3 agent representation.\n');
agent = rlEngine('create_td3', cfg);

end

%% Helper: Critic Network Builder for MATLAB RL Toolbox
function critic = buildCriticNet(id_str, obs_dim, act_dim, cfg, obs_info, act_info)
    state_path = [
        featureInputLayer(obs_dim, 'Normalization', 'none', 'Name', ['state_in_' id_str])
        fullyConnectedLayer(cfg.hidden_units(1), 'Name', ['crit_sfc_' id_str])
    ];
    action_path = [
        featureInputLayer(act_dim, 'Normalization', 'none', 'Name', ['act_in_' id_str])
        fullyConnectedLayer(cfg.hidden_units(1), 'Name', ['crit_afc_' id_str])
    ];
    common_path = [
        additionLayer(2, 'Name', ['add_' id_str])
        reluLayer('Name', ['relu1_' id_str])
        fullyConnectedLayer(cfg.hidden_units(2), 'Name', ['crit_fc2_' id_str])
        reluLayer('Name', ['relu2_' id_str])
        fullyConnectedLayer(1, 'Name', ['q_out_' id_str])
    ];
    
    lg = layerGraph(state_path);
    lg = addLayers(lg, action_path);
    lg = addLayers(lg, common_path);
    lg = connectLayers(lg, ['crit_sfc_' id_str], ['add_' id_str '/in1']);
    lg = connectLayers(lg, ['crit_afc_' id_str], ['add_' id_str '/in2']);
    
    crit_net = dlnetwork(lg);
    critic = rlQValueFunction(crit_net, obs_info, act_info, ...
        'ObservationInputNames', ['state_in_' id_str], ...
        'ActionInputNames', ['act_in_' id_str]);
end
