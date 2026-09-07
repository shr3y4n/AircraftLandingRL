function varargout = rlEngine(cmd, varargin)
% RLENGINE Standalone pure-MATLAB continuous reinforcement learning engine.
%
% Implements forward inference, replay buffer management, stochastic policy
% sampling, and Adam-based gradient updates for continuous actor-critic agents:
%   - TD3: Twin Delayed Deep Deterministic Policy Gradient
%   - SAC: Soft Actor-Critic with maximum-entropy reinforcement learning
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

% 1. Actor Architecture
if strcmp(agent.type, 'sac')
    % SAC Stochastic Actor: outputs mean (1:act_dim) and log_std (act_dim+1:2*act_dim)
    agent.actor.W1 = randn(h1, obs_dim) * sqrt(2.0 / obs_dim);
    agent.actor.b1 = zeros(h1, 1);
    agent.actor.W2 = randn(h2, h1) * sqrt(2.0 / h1);
    agent.actor.b2 = zeros(h2, 1);
    agent.actor.W3 = randn(2 * act_dim, h2) * sqrt(2.0 / h2) * 0.1;
    agent.actor.b3 = zeros(2 * act_dim, 1);
    
    % SAC Entropy Temperature Parameters
    agent.log_alpha      = log(cfg.init_alpha);
    agent.alpha          = cfg.init_alpha;
    agent.target_entropy = cfg.entropy_target; % -act_dim = -2
else
    % TD3 Deterministic Actor: outputs action in [-1, +1]
    agent.actor.W1 = randn(h1, obs_dim) * sqrt(2.0 / obs_dim);
    agent.actor.b1 = zeros(h1, 1);
    agent.actor.W2 = randn(h2, h1) * sqrt(2.0 / h1);
    agent.actor.b2 = zeros(h2, 1);
    agent.actor.W3 = randn(act_dim, h2) * sqrt(2.0 / h2);
    agent.actor.b3 = zeros(act_dim, 1);
    agent.target_actor = agent.actor;
end

% 2. Twin Q-Critics: Q1(s, a), Q2(s, a)
crit_in = obs_dim + act_dim;
agent.critic1.W1 = randn(h1, crit_in) * sqrt(2.0 / crit_in);
agent.critic1.b1 = zeros(h1, 1);
agent.critic1.W2 = randn(h2, h1) * sqrt(2.0 / h1);
agent.critic1.b2 = zeros(h2, 1);
agent.critic1.W3 = randn(1, h2) * sqrt(2.0 / h2);
agent.critic1.b3 = zeros(1, 1);
agent.target_critic1 = agent.critic1;

agent.critic2.W1 = randn(h1, crit_in) * sqrt(2.0 / crit_in);
agent.critic2.b1 = zeros(h1, 1);
agent.critic2.W2 = randn(h2, h1) * sqrt(2.0 / h1);
agent.critic2.b2 = zeros(h2, 1);
agent.critic2.W3 = randn(1, h2) * sqrt(2.0 / h2);
agent.critic2.b3 = zeros(1, 1);
agent.target_critic2 = agent.critic2;

% 3. Experience Replay Buffer
cap = cfg.buffer_capacity;
agent.buffer.capacity = cap;
agent.buffer.size     = 0;
agent.buffer.ptr      = 1;
agent.buffer.obs      = zeros(obs_dim, cap);
agent.buffer.act      = zeros(act_dim, cap);
agent.buffer.rew      = zeros(1, cap);
agent.buffer.next_obs = zeros(obs_dim, cap);
agent.buffer.done     = zeros(1, cap);

% 4. Optimizer Momentum / Adam Cache
agent.adam = initAdam(agent);

end

%% Sub-function: Forward Policy Inference
function action = getAction(agent, obs, add_noise)
if nargin < 3, add_noise = false; end
obs = obs(:);

if strcmp(agent.type, 'sac')
    act_dim = agent.cfg.act_dim;
    z1 = agent.actor.W1 * obs + agent.actor.b1;
    a1 = max(0.0, z1);
    z2 = agent.actor.W2 * a1 + agent.actor.b2;
    a2 = max(0.0, z2);
    z3 = agent.actor.W3 * a2 + agent.actor.b3;
    mu = z3(1:act_dim, :);
    
    if add_noise
        log_std = min(2.0, max(-20.0, z3(act_dim+1:end, :)));
        std_dev = exp(log_std);
        u = mu + std_dev .* randn(size(mu));
        action = tanh(u);
    else
        action = tanh(mu);
    end
else
    % TD3
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

%% Sub-function: Mini-Batch Gradient Update Dispatcher
function [agent, loss_info] = updateAgent(agent)
if strcmp(agent.type, 'sac')
    [agent, loss_info] = updateSAC(agent);
else
    [agent, loss_info] = updateTD3(agent);
end
end

%% Sub-function: TD3 Mini-Batch Gradient Update
function [agent, loss_info] = updateTD3(agent)
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

% 2. Compute Target Actions with Smoothing Noise
a_target = forwardTD3ActorBatch(agent.target_actor, s2_b);
target_noise = min(cfg.noise_clip, max(-cfg.noise_clip, randn(size(a_target)) * cfg.target_noise));
a_target = min(1.0, max(-1.0, a_target + target_noise));

% 3. Compute Bellman Target Q-Values: y = r + gamma * (1 - done) * min(Q1_target, Q2_target)
sa_target = [s2_b; a_target];
q1_tgt = forwardCriticBatch(agent.target_critic1, sa_target);
q2_tgt = forwardCriticBatch(agent.target_critic2, sa_target);
q_target_min = min(q1_tgt, q2_tgt);

y_target = r_b + cfg.gamma * (1.0 - done_b) .* q_target_min;

% 4. Compute Critic Gradients and Update
sa_cur = [s_b; a_b];
[q1_pred, q1_cache] = forwardCriticBatch(agent.critic1, sa_cur);
[q2_pred, q2_cache] = forwardCriticBatch(agent.critic2, sa_cur);

err1 = q1_pred - y_target;
err2 = q2_pred - y_target;

loss_info.critic_loss = 0.5 * mean(err1.^2 + err2.^2);

grad_q1 = backwardCriticBatch(agent.critic1, err1, q1_cache);
grad_q2 = backwardCriticBatch(agent.critic2, err2, q2_cache);

[agent.critic1, agent.adam.m_q1, agent.adam.v_q1] = adamStepDirect(agent.critic1, grad_q1, agent.adam.m_q1, agent.adam.v_q1, cfg.critic_lr, agent.step_total);
[agent.critic2, agent.adam.m_q2, agent.adam.v_q2] = adamStepDirect(agent.critic2, grad_q2, agent.adam.m_q2, agent.adam.v_q2, cfg.critic_lr, agent.step_total);

% 5. Delayed Policy Update (every policy_delay steps)
if mod(agent.step_total, cfg.policy_delay) == 0
    [a_cur, act_cache] = forwardTD3ActorBatch(agent.actor, s_b);
    sa_policy = [s_b; a_cur];
    
    grad_q_wrt_a = criticActionGradient(agent.critic1, sa_policy, cfg.obs_dim);
    grad_actor = backwardTD3ActorBatch(agent.actor, -grad_q_wrt_a, act_cache);
    
    [agent.actor, agent.adam.m_act, agent.adam.v_act] = adamStepDirect(agent.actor, grad_actor, agent.adam.m_act, agent.adam.v_act, cfg.actor_lr, agent.step_total);
    loss_info.actor_loss = -mean(forwardCriticBatch(agent.critic1, sa_policy));
    
    % Polyak Soft Updates
    agent.target_actor   = softUpdateDirect(agent.actor,   agent.target_actor,   cfg.tau);
    agent.target_critic1 = softUpdateDirect(agent.critic1, agent.target_critic1, cfg.tau);
    agent.target_critic2 = softUpdateDirect(agent.critic2, agent.target_critic2, cfg.tau);
end

end

%% Sub-function: SAC Mini-Batch Gradient Update
function [agent, loss_info] = updateSAC(agent)
loss_info.critic_loss = 0.0;
loss_info.actor_loss  = 0.0;

if agent.buffer.size < agent.cfg.batch_size
    return;
end

cfg     = agent.cfg;
B       = cfg.batch_size;
act_dim = cfg.act_dim;

% 1. Sample Random Mini-batch
idx    = randi(agent.buffer.size, [B, 1]);
s_b    = agent.buffer.obs(:, idx);
a_b    = agent.buffer.act(:, idx);
r_b    = agent.buffer.rew(idx);
s2_b   = agent.buffer.next_obs(:, idx);
done_b = agent.buffer.done(idx);

% 2. Sample Next Action from Stochastic Policy pi(s')
[a_next, logp_next] = sampleSACPolicy(agent.actor, s2_b, act_dim);

% 3. Soft Bellman Target: y = r + gamma * (1 - done) * (min(Q1_tgt, Q2_tgt) - alpha * logp_next)
sa_next = [s2_b; a_next];
q1_tgt = forwardCriticBatch(agent.target_critic1, sa_next);
q2_tgt = forwardCriticBatch(agent.target_critic2, sa_next);
q_tgt_min = min(q1_tgt, q2_tgt);

y_target = r_b + cfg.gamma * (1.0 - done_b) .* (q_tgt_min - agent.alpha * logp_next);

% 4. Critic Gradients & Direct Adam Optimization
sa_cur = [s_b; a_b];
[q1_pred, q1_cache] = forwardCriticBatch(agent.critic1, sa_cur);
[q2_pred, q2_cache] = forwardCriticBatch(agent.critic2, sa_cur);

err1 = q1_pred - y_target;
err2 = q2_pred - y_target;

loss_info.critic_loss = 0.5 * mean(err1.^2 + err2.^2);

grad_q1 = backwardCriticBatch(agent.critic1, err1, q1_cache);
grad_q2 = backwardCriticBatch(agent.critic2, err2, q2_cache);

[agent.critic1, agent.adam.m_q1, agent.adam.v_q1] = adamStepDirect(agent.critic1, grad_q1, agent.adam.m_q1, agent.adam.v_q1, cfg.critic_lr, agent.step_total);
[agent.critic2, agent.adam.m_q2, agent.adam.v_q2] = adamStepDirect(agent.critic2, grad_q2, agent.adam.m_q2, agent.adam.v_q2, cfg.critic_lr, agent.step_total);

% 5. Actor Policy Gradient Update (Reparameterized Action Gradient)
[a_curr, logp_curr, act_cache, std_curr, eps_curr] = sampleSACPolicy(agent.actor, s_b, act_dim);
sa_curr = [s_b; a_curr];

q1_pi = forwardCriticBatch(agent.critic1, sa_curr);
q2_pi = forwardCriticBatch(agent.critic2, sa_curr);

grad_q1_wrt_a = criticActionGradient(agent.critic1, sa_curr, cfg.obs_dim);
grad_q2_wrt_a = criticActionGradient(agent.critic2, sa_curr, cfg.obs_dim);
use_q1 = (q1_pi <= q2_pi);
grad_q_wrt_a = grad_q1_wrt_a .* use_q1 + grad_q2_wrt_a .* (~use_q1);

% Closed-form gradient w.r.t mean and log_std
grad_mu = -grad_q_wrt_a .* (1.0 - a_curr.^2) + agent.alpha * (2.0 * a_curr);
grad_logstd = -grad_q_wrt_a .* (1.0 - a_curr.^2) .* (std_curr .* eps_curr) + ...
              agent.alpha * (-1.0 + 2.0 * a_curr .* (std_curr .* eps_curr));
grad_z3 = min(5.0, max(-5.0, [grad_mu; grad_logstd]));

grad_actor = backwardSACActorBatch(agent.actor, grad_z3, act_cache);
[agent.actor, agent.adam.m_act, agent.adam.v_act] = adamStepDirect(agent.actor, grad_actor, agent.adam.m_act, agent.adam.v_act, cfg.actor_lr, agent.step_total);

loss_info.actor_loss = mean(agent.alpha * logp_curr - min(q1_pi, q2_pi));

% 6. Automatic Entropy Temperature Adjustment
grad_log_alpha = -mean(logp_curr + agent.target_entropy);
grad_log_alpha = min(5.0, max(-5.0, grad_log_alpha));
[agent.log_alpha, agent.adam.m_alpha, agent.adam.v_alpha] = adamScalarStep(...
    agent.log_alpha, grad_log_alpha, agent.adam.m_alpha, agent.adam.v_alpha, cfg.alpha_lr, agent.step_total);
agent.alpha = min(1.0, max(0.01, exp(agent.log_alpha)));

% 7. Polyak Target Network Soft Updates for Critics
agent.target_critic1 = softUpdateDirect(agent.critic1, agent.target_critic1, cfg.tau);
agent.target_critic2 = softUpdateDirect(agent.critic2, agent.target_critic2, cfg.tau);

end

%% Helper: Neural Network Forward / Backward Kernels
function [act, cache] = forwardTD3ActorBatch(net, obs_b)
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

function [a, logp, cache, std_dev, eps_noise] = sampleSACPolicy(net, obs_b, act_dim)
z1 = net.W1 * obs_b + net.b1;
a1 = max(0.0, z1);
z2 = net.W2 * a1 + net.b2;
a2 = max(0.0, z2);
z3 = net.W3 * a2 + net.b3;

mu      = z3(1:act_dim, :);
log_std = min(2.0, max(-20.0, z3(act_dim+1:end, :)));
std_dev = exp(log_std);

eps_noise = randn(size(mu));
u = mu + std_dev .* eps_noise;
a = tanh(u);

% Log probability with tanh squashing correction
log_gauss   = -0.5 * sum(eps_noise.^2 + 2.0 * log_std + log(2.0 * pi), 1);
squash_corr = sum(log(1.0 - a.^2 + 1e-6), 1);
logp        = log_gauss - squash_corr;

cache.obs = obs_b;
cache.z1  = z1;
cache.a1  = a1;
cache.z2  = z2;
cache.a2  = a2;
cache.z3  = z3;
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

function grads = backwardTD3ActorBatch(net, dL_da, cache)
B = size(dL_da, 2);
dL_dz3 = dL_da .* (1.0 - cache.act.^2);
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

function grads = backwardSACActorBatch(net, grad_z3, cache)
B = size(grad_z3, 2);
dL_da2 = net.W3' * grad_z3;
dL_dz2 = dL_da2 .* (cache.z2 > 0);
dL_da1 = net.W2' * dL_dz2;
dL_dz1 = dL_da1 .* (cache.z1 > 0);

grads.W3 = (grad_z3 * cache.a2') / B;
grads.b3 = sum(grad_z3, 2) / B;
grads.W2 = (dL_dz2 * cache.a1') / B;
grads.b2 = sum(dL_dz2, 2) / B;
grads.W1 = (dL_dz1 * cache.obs') / B;
grads.b1 = sum(dL_dz1, 2) / B;
end

%% Helper: Fast Direct Adam Optimizer & Target Updates
function adam = initAdam(agent)
function p = initParamAdam(net)
    p.m.W1 = zeros(size(net.W1)); p.v.W1 = zeros(size(net.W1));
    p.m.b1 = zeros(size(net.b1)); p.v.b1 = zeros(size(net.b1));
    p.m.W2 = zeros(size(net.W2)); p.v.W2 = zeros(size(net.W2));
    p.m.b2 = zeros(size(net.b2)); p.v.b2 = zeros(size(net.b2));
    p.m.W3 = zeros(size(net.W3)); p.v.W3 = zeros(size(net.W3));
    p.m.b3 = zeros(size(net.b3)); p.v.b3 = zeros(size(net.b3));
end
adam.m_act = initParamAdam(agent.actor).m;
adam.v_act = initParamAdam(agent.actor).v;
adam.m_q1  = initParamAdam(agent.critic1).m;
adam.v_q1  = initParamAdam(agent.critic1).v;
adam.m_q2  = initParamAdam(agent.critic2).m;
adam.v_q2  = initParamAdam(agent.critic2).v;
adam.m_alpha = 0.0;
adam.v_alpha = 0.0;
end

function [net, m, v] = adamStepDirect(net, grads, m, v, lr, t)
beta1 = 0.9;
beta2 = 0.999;
eps   = 1e-8;
t = max(1, t);

corr1 = 1.0 - beta1^t;
corr2 = 1.0 - beta2^t;
lr_eff = lr * sqrt(corr2) / corr1;
eps_eff = eps * sqrt(corr2);

% W1, b1
m.W1 = beta1 * m.W1 + (1.0 - beta1) * grads.W1;
v.W1 = beta2 * v.W1 + (1.0 - beta2) * (grads.W1.^2);
net.W1 = net.W1 - lr_eff * (m.W1 ./ (sqrt(v.W1) + eps_eff));

m.b1 = beta1 * m.b1 + (1.0 - beta1) * grads.b1;
v.b1 = beta2 * v.b1 + (1.0 - beta2) * (grads.b1.^2);
net.b1 = net.b1 - lr_eff * (m.b1 ./ (sqrt(v.b1) + eps_eff));

% W2, b2
m.W2 = beta1 * m.W2 + (1.0 - beta1) * grads.W2;
v.W2 = beta2 * v.W2 + (1.0 - beta2) * (grads.W2.^2);
net.W2 = net.W2 - lr_eff * (m.W2 ./ (sqrt(v.W2) + eps_eff));

m.b2 = beta1 * m.b2 + (1.0 - beta1) * grads.b2;
v.b2 = beta2 * v.b2 + (1.0 - beta2) * (grads.b2.^2);
net.b2 = net.b2 - lr_eff * (m.b2 ./ (sqrt(v.b2) + eps_eff));

% W3, b3
m.W3 = beta1 * m.W3 + (1.0 - beta1) * grads.W3;
v.W3 = beta2 * v.W3 + (1.0 - beta2) * (grads.W3.^2);
net.W3 = net.W3 - lr_eff * (m.W3 ./ (sqrt(v.W3) + eps_eff));

m.b3 = beta1 * m.b3 + (1.0 - beta1) * grads.b3;
v.b3 = beta2 * v.b3 + (1.0 - beta2) * (grads.b3.^2);
net.b3 = net.b3 - lr_eff * (m.b3 ./ (sqrt(v.b3) + eps_eff));
end

function [param, m, v] = adamScalarStep(param, grad, m, v, lr, t)
beta1 = 0.9;
beta2 = 0.999;
eps   = 1e-8;
t = max(1, t);

m = beta1 * m + (1.0 - beta1) * grad;
v = beta2 * v + (1.0 - beta2) * (grad.^2);

m_hat = m / (1.0 - beta1^t);
v_hat = v / (1.0 - beta2^t);

param = param - lr * (m_hat / (sqrt(v_hat) + eps));
end

function tgt = softUpdateDirect(src, tgt, tau)
one_minus_tau = 1.0 - tau;
tgt.W1 = tau * src.W1 + one_minus_tau * tgt.W1;
tgt.b1 = tau * src.b1 + one_minus_tau * tgt.b1;
tgt.W2 = tau * src.W2 + one_minus_tau * tgt.W2;
tgt.b2 = tau * src.b2 + one_minus_tau * tgt.b2;
tgt.W3 = tau * src.W3 + one_minus_tau * tgt.W3;
tgt.b3 = tau * src.b3 + one_minus_tau * tgt.b3;
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
