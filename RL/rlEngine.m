function varargout = rlEngine(cmd, varargin)
% RLENGINE Standalone pure-MATLAB deep continuous reinforcement learning engine.
%
% Implements forward inference, replay buffer management, and policy execution
% for continuous actor-critic agents (TD3/SAC) without requiring external toolboxes.
%
% Supported subcommands:
%   agent = rlEngine('create_td3', cfg)
%   agent = rlEngine('create_sac', cfg)
%   action = rlEngine('getAction', agent, obs, add_noise)
%   agent = rlEngine('storeTransition', agent, s, a, r, s_next, done)
%   [agent, loss] = rlEngine('update', agent)
%   rlEngine('save', agent, filepath)
%   agent = rlEngine('load', filepath)

switch lower(cmd)
    case 'create_td3'
        varargout{1} = createAgent('td3', varargin{:});
    case 'create_sac'
        varargout{1} = createAgent('sac', varargin{:});
    case 'getaction'
        varargout{1} = getAction(varargin{:});
    case 'storetransition'
        varargout{1} = storeTransition(varargin{:});
    case 'update'
        [varargout{1}, varargout{2}] = updateAgent(varargin{:});
    case 'save'
        saveAgent(varargin{:});
    case 'load'
        varargout{1} = loadAgent(varargin{:});
    otherwise
        error('rlEngine:UnknownCommand', 'Unknown command: %s', cmd);
end

end

%% Sub-function: Agent Constructor
function agent = createAgent(type, cfg)
if nargin < 2 || isempty(cfg), cfg = rlConfig(); end

agent.type = lower(type);
agent.cfg  = cfg;
agent.step_total = 0;

obs_dim = cfg.obs_dim;
act_dim = cfg.act_dim;
h1 = cfg.hidden_units(1);
h2 = cfg.hidden_units(2);

% Xavier / He weight initialization
agent.actor.W1 = randn(h1, obs_dim) * sqrt(2.0 / obs_dim);
agent.actor.b1 = zeros(h1, 1);
agent.actor.W2 = randn(h2, h1) * sqrt(2.0 / h1);
agent.actor.b2 = zeros(h2, 1);
agent.actor.W3 = randn(act_dim, h2) * sqrt(2.0 / h2);
agent.actor.b3 = zeros(act_dim, 1);

% Target Actor (initialized with identical weights)
agent.target_actor = agent.actor;

% Critic 1: Q1(s, a)
crit_in = obs_dim + act_dim;
agent.critic1.W1 = randn(h1, crit_in) * sqrt(2.0 / crit_in);
agent.critic1.b1 = zeros(h1, 1);
agent.critic1.W2 = randn(h2, h1) * sqrt(2.0 / h1);
agent.critic1.b2 = zeros(h2, 1);
agent.critic1.W3 = randn(1, h2) * sqrt(2.0 / h2);
agent.critic1.b3 = zeros(1, 1);
agent.target_critic1 = agent.critic1;

% Critic 2: Q2(s, a)
agent.critic2.W1 = randn(h1, crit_in) * sqrt(2.0 / crit_in);
agent.critic2.b1 = zeros(h1, 1);
agent.critic2.W2 = randn(h2, h1) * sqrt(2.0 / h1);
agent.critic2.b2 = zeros(h2, 1);
agent.critic2.W3 = randn(1, h2) * sqrt(2.0 / h2);
agent.critic2.b3 = zeros(1, 1);
agent.target_critic2 = agent.critic2;

% Circular Experience Replay Buffer
cap = cfg.buffer_capacity;
agent.buffer.capacity = cap;
agent.buffer.size     = 0;
agent.buffer.ptr      = 1;
agent.buffer.obs      = zeros(obs_dim, cap);
agent.buffer.act      = zeros(act_dim, cap);
agent.buffer.rew      = zeros(1, cap);
agent.buffer.next_obs = zeros(obs_dim, cap);
agent.buffer.done     = zeros(1, cap);

% Optimizer Momentum / Adam Cache
agent.adam = initAdam(agent);

end

%% Sub-function: Forward Policy Inference
function action = getAction(agent, obs, add_noise)
if nargin < 3, add_noise = false; end
obs = obs(:);

% MLP Forward: Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh
z1 = agent.actor.W1 * obs + agent.actor.b1;
a1 = max(0.0, z1);
z2 = agent.actor.W2 * a1 + agent.actor.b2;
a2 = max(0.0, z2);
z3 = agent.actor.W3 * a2 + agent.actor.b3;
action = tanh(z3);

if add_noise
    noise = randn(size(action)) * agent.cfg.noise_sigma;
    action = min(1.0, max(-1.0, action + noise));
end

end

%% Sub-function: Store Experience in Circular Buffer
function agent = storeTransition(agent, s, a, r, s_next, done)
ptr = agent.buffer.ptr;

agent.buffer.obs(:, ptr)      = s(:);
agent.buffer.act(:, ptr)      = a(:);
agent.buffer.rew(ptr)         = r;
agent.buffer.next_obs(:, ptr) = s_next(:);
agent.buffer.done(ptr)        = double(done);

agent.buffer.size = min(agent.buffer.capacity, agent.buffer.size + 1);
agent.buffer.ptr  = mod(ptr, agent.buffer.capacity) + 1;
agent.step_total  = agent.step_total + 1;
end

%% Sub-function: TD3 Mini-Batch Gradient Update
function [agent, loss_info] = updateAgent(agent)
loss_info.critic_loss = 0.0;
loss_info.actor_loss  = 0.0;

if agent.buffer.size < agent.cfg.batch_size
    return;
end

cfg = agent.cfg;
B   = cfg.batch_size;

% 1. Sample Random Mini-batch
idx = randi(agent.buffer.size, [B, 1]);
s_b    = agent.buffer.obs(:, idx);
a_b    = agent.buffer.act(:, idx);
r_b    = agent.buffer.rew(idx);
s2_b   = agent.buffer.next_obs(:, idx);
done_b = agent.buffer.done(idx);

% 2. Compute Target Actions with Target Policy Smoothing Noise
% a'(s') = clip(target_actor(s') + clip(noise, -c, +c), -1, +1)
a_target = forwardActorBatch(agent.target_actor, s2_b);
target_noise = min(cfg.noise_clip, max(-cfg.noise_clip, randn(size(a_target)) * cfg.target_noise));
a_target = min(1.0, max(-1.0, a_target + target_noise));

% 3. Compute Bellman Target Q-Values: y = r + gamma * (1 - done) * min(Q1_target, Q2_target)
sa_target = [s2_b; a_target];
q1_tgt = forwardCriticBatch(agent.target_critic1, sa_target);
q2_tgt = forwardCriticBatch(agent.target_critic2, sa_target);
q_target_min = min(q1_tgt, q2_tgt);

y_target = r_b + cfg.gamma * (1.0 - done_b) .* q_target_min;

% 4. Compute Critic Gradients and Update via Adam
sa_cur = [s_b; a_b];
[q1_pred, q1_cache] = forwardCriticBatch(agent.critic1, sa_cur);
[q2_pred, q2_cache] = forwardCriticBatch(agent.critic2, sa_cur);

err1 = q1_pred - y_target;
err2 = q2_pred - y_target;

loss_info.critic_loss = 0.5 * mean(err1.^2 + err2.^2);

% Backward pass on critics
grad_q1 = backwardCriticBatch(agent.critic1, err1, q1_cache);
grad_q2 = backwardCriticBatch(agent.critic2, err2, q2_cache);

[agent.critic1, agent.adam.m_q1, agent.adam.v_q1] = adamStep(agent.critic1, grad_q1, agent.adam.m_q1, agent.adam.v_q1, cfg.critic_lr, agent.step_total);
[agent.critic2, agent.adam.m_q2, agent.adam.v_q2] = adamStep(agent.critic2, grad_q2, agent.adam.m_q2, agent.adam.v_q2, cfg.critic_lr, agent.step_total);

% 5. Delayed Policy Update (every policy_delay steps)
if mod(agent.step_total, cfg.policy_delay) == 0
    [a_cur, act_cache] = forwardActorBatch(agent.actor, s_b);
    sa_policy = [s_b; a_cur];
    
    % Gradient of Q1 with respect to action a
    grad_q_wrt_a = criticActionGradient(agent.critic1, sa_policy, cfg.obs_dim);
    
    % Deterministic policy gradient: grad_theta = grad_a Q * grad_theta mu
    grad_actor = backwardActorBatch(agent.actor, -grad_q_wrt_a, act_cache);
    
    [agent.actor, agent.adam.m_act, agent.adam.v_act] = adamStep(agent.actor, grad_actor, agent.adam.m_act, agent.adam.v_act, cfg.actor_lr, agent.step_total);
    loss_info.actor_loss = -mean(forwardCriticBatch(agent.critic1, sa_policy));
    
    % Polyak Target Network Soft Updates: theta_target = tau * theta + (1 - tau) * theta_target
    agent.target_actor   = softUpdate(agent.actor,   agent.target_actor,   cfg.tau);
    agent.target_critic1 = softUpdate(agent.critic1, agent.target_critic1, cfg.tau);
    agent.target_critic2 = softUpdate(agent.critic2, agent.target_critic2, cfg.tau);
end

end

%% Helper: Neural Network Forward / Backward Kernels
function [act, cache] = forwardActorBatch(net, obs_b)
z1 = net.W1 * obs_b + net.b1;
a1 = max(0.0, z1);
z2 = net.W2 * a1 + net.b2;
a2 = max(0.0, z2);
z3 = net.W3 * a2 + net.b3;
act = tanh(z3);

cache.obs = obs_b;
cache.z1  = z1;
cache.a1  = a1;
cache.z2  = z2;
cache.a2  = a2;
cache.z3  = z3;
cache.act = act;
end

function [q, cache] = forwardCriticBatch(net, in_b)
z1 = net.W1 * in_b + net.b1;
a1 = max(0.0, z1);
z2 = net.W2 * a1 + net.b2;
a2 = max(0.0, z2);
q  = net.W3 * a2 + net.b3;

cache.in = in_b;
cache.z1 = z1;
cache.a1 = a1;
cache.z2 = z2;
cache.a2 = a2;
end

function grads = backwardCriticBatch(net, dL_dq, cache)
B = length(dL_dq);
dL_da2 = net.W3' * dL_dq;
dL_dz2 = dL_da2 .* (cache.z2 > 0);
dL_da1 = net.W2' * dL_dz2;
dL_dz1 = dL_da1 .* (cache.z1 > 0);

grads.W3 = (dL_dq * cache.a2') / B;
grads.b3 = sum(dL_dq, 2) / B;
grads.W2 = (dL_dz2 * cache.a1') / B;
grads.b2 = sum(dL_dz2, 2) / B;
grads.W1 = (dL_dz1 * cache.in') / B;
grads.b1 = sum(dL_dz1, 2) / B;
end

function grad_a = criticActionGradient(net, sa_b, obs_dim)
z1 = net.W1 * sa_b + net.b1;
a1 = max(0.0, z1);
z2 = net.W2 * a1 + net.b2;
a2 = max(0.0, z2);

dL_dq  = ones(1, size(sa_b, 2));
dL_da2 = net.W3' * dL_dq;
dL_dz2 = dL_da2 .* (z2 > 0);
dL_da1 = net.W2' * dL_dz2;
dL_dz1 = dL_da1 .* (z1 > 0);
dL_din = net.W1' * dL_dz1;

% Extract action gradient portion
grad_a = dL_din(obs_dim+1:end, :);
end

function grads = backwardActorBatch(net, dL_da, cache)
B = size(dL_da, 2);
dL_dz3 = dL_da .* (1.0 - cache.act.^2); % d/dz tanh(z) = 1 - tanh^2
dL_da2 = net.W3' * dL_dz3;
dL_dz2 = dL_da2 .* (cache.z2 > 0);
dL_da1 = net.W2' * dL_dz2;
dL_dz1 = dL_da1 .* (cache.z1 > 0);

grads.W3 = (dL_dz3 * cache.a2') / B;
grads.b3 = sum(dL_dz3, 2) / B;
grads.W2 = (dL_dz2 * cache.a1') / B;
grads.b2 = sum(dL_dz2, 2) / B;
grads.W1 = (dL_dz1 * cache.obs') / B;
grads.b1 = sum(dL_dz1, 2) / B;
end

%% Helper: Adam Optimizer & Target Updates
function adam = initAdam(agent)
function p = initParamAdam(net)
    flds = fieldnames(net);
    for k = 1:length(flds)
        p.m.(flds{k}) = zeros(size(net.(flds{k})));
        p.v.(flds{k}) = zeros(size(net.(flds{k})));
    end
end
adam.m_act = initParamAdam(agent.actor).m;
adam.v_act = initParamAdam(agent.actor).v;
adam.m_q1  = initParamAdam(agent.critic1).m;
adam.v_q1  = initParamAdam(agent.critic1).v;
adam.m_q2  = initParamAdam(agent.critic2).m;
adam.v_q2  = initParamAdam(agent.critic2).v;
end

function [net, m, v] = adamStep(net, grads, m, v, lr, t)
beta1 = 0.9;
beta2 = 0.999;
eps   = 1e-8;
t = max(1, t);

flds = fieldnames(grads);
for k = 1:length(flds)
    f = flds{k};
    g = grads.(f);
    m.(f) = beta1 * m.(f) + (1.0 - beta1) * g;
    v.(f) = beta2 * v.(f) + (1.0 - beta2) * (g.^2);
    
    m_hat = m.(f) / (1.0 - beta1^t);
    v_hat = v.(f) / (1.0 - beta2^t);
    
    net.(f) = net.(f) - lr * (m_hat ./ (sqrt(v_hat) + eps));
end
end

function tgt = softUpdate(src, tgt, tau)
flds = fieldnames(src);
for k = 1:length(flds)
    f = flds{k};
    tgt.(f) = tau * src.(f) + (1.0 - tau) * tgt.(f);
end
end

%% Helper: Save and Load
function saveAgent(agent, filepath)
save(filepath, 'agent');
fprintf('[RL Engine] Successfully saved policy checkpoint to: %s\n', filepath);
end

function agent = loadAgent(filepath)
data = load(filepath);
agent = data.agent;
fprintf('[RL Engine] Successfully loaded policy checkpoint from: %s\n', filepath);
end
