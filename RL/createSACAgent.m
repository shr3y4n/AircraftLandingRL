function agent = createSACAgent(obs_info, act_info, cfg)
% CREATESACAGENT Constructs a Soft Actor-Critic (SAC) continuous reinforcement learning agent.
%
% Implements maximum-entropy reinforcement learning with twin Q-critics and
% automatic entropy temperature parameter adjustment.
%
% Inputs:
%   obs_info - Observation specification from createEnvironment()
%   act_info - Action specification from createEnvironment()
%   cfg      - Hyperparameters struct from rlConfig()
%
% Outputs:
%   agent - Configured SAC agent object or standalone fallback agent

if nargin < 3 || isempty(cfg)
    cfg = rlConfig();
end

obs_dim = cfg.obs_dim;
act_dim = cfg.act_dim;

has_rl_toolbox = (exist('rlSACAgent', 'file') == 2 || exist('rlSACAgent', 'class') == 8);

if has_rl_toolbox
    try
        fprintf('[SAC Setup] Initializing MATLAB RL Toolbox SAC networks...\n');
        
        % 1. STOCHASTIC GAUSSIAN ACTOR NETWORK
        % Outputs mean and standard deviation for squashed Gaussian distribution
        actor_layers = [
            featureInputLayer(obs_dim, 'Normalization', 'none', 'Name', 'obs_in')
            fullyConnectedLayer(cfg.hidden_units(1), 'Name', 'act_fc1')
            reluLayer('Name', 'act_relu1')
            fullyConnectedLayer(cfg.hidden_units(2), 'Name', 'act_fc2')
            reluLayer('Name', 'act_relu2')
            fullyConnectedLayer(act_dim * 2, 'Name', 'mean_and_logstd')
        ];
        actor_net = dlnetwork(layerGraph(actor_layers));
        actor = rlContinuousGaussianActor(actor_net, obs_info, act_info, ...
            'ObservationInputNames', 'obs_in', 'ActionMeanOutputNames', 'mean_and_logstd');

        critic1 = buildSACCritic('q1', obs_dim, act_dim, cfg, obs_info, act_info);
        critic2 = buildSACCritic('q2', obs_dim, act_dim, cfg, obs_info, act_info);

        % 3. SAC AGENT OPTIONS
        agent_opts = rlSACAgentOptions();
        agent_opts.SampleTime             = cfg.sample_time;
        agent_opts.DiscountFactor         = cfg.gamma;
        agent_opts.TargetSmoothFactor     = cfg.tau;
        agent_opts.ExperienceBufferLength = cfg.buffer_capacity;
        agent_opts.MiniBatchSize          = cfg.batch_size;
        agent_opts.EntropyWeightOptions.TargetEntropy = cfg.entropy_target;

        agent = rlSACAgent(actor, [critic1, critic2], agent_opts);
        fprintf('[SAC Setup] Successfully created official rlSACAgent.\n');
        return;
    catch ME
        fprintf('[SAC Setup] RL Toolbox notice: %s. Using standalone engine.\n', ME.message);
    end
end

% Fallback: Standalone Pure-MATLAB SAC Policy Agent
fprintf('[SAC Setup] Initializing standalone pure-MATLAB SAC agent representation.\n');
agent = rlEngine('create_sac', cfg);

end

%% Helper: Critic Network Builder for MATLAB RL Toolbox
function critic = buildSACCritic(id_str, obs_dim, act_dim, cfg, obs_info, act_info)
    state_path = [
        featureInputLayer(obs_dim, 'Normalization', 'none', 'Name', ['s_in_' id_str])
        fullyConnectedLayer(cfg.hidden_units(1), 'Name', ['s_fc_' id_str])
    ];
    action_path = [
        featureInputLayer(act_dim, 'Normalization', 'none', 'Name', ['a_in_' id_str])
        fullyConnectedLayer(cfg.hidden_units(1), 'Name', ['a_fc_' id_str])
    ];
    common_path = [
        additionLayer(2, 'Name', ['add_' id_str])
        reluLayer('Name', ['relu1_' id_str])
        fullyConnectedLayer(cfg.hidden_units(2), 'Name', ['fc2_' id_str])
        reluLayer('Name', ['relu2_' id_str])
        fullyConnectedLayer(1, 'Name', ['q_out_' id_str])
    ];
    
    lg = layerGraph(state_path);
    lg = addLayers(lg, action_path);
    lg = addLayers(lg, common_path);
    lg = connectLayers(lg, ['s_fc_' id_str], ['add_' id_str '/in1']);
    lg = connectLayers(lg, ['a_fc_' id_str], ['add_' id_str '/in2']);
    
    crit_net = dlnetwork(lg);
    critic = rlQValueFunction(crit_net, obs_info, act_info, ...
        'ObservationInputNames', ['s_in_' id_str], ...
        'ActionInputNames', ['a_in_' id_str]);
end
