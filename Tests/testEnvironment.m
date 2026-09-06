function results = testEnvironment()
% TESTENVIRONMENT Unit tests for the reinforcement learning environment, action scaling, and reward.

fprintf('\n>>> Running RL Environment & Reward Function Tests...\n');

P = aircraftParameters();
results.passed = 0;
results.failed = 0;

%% Test 1: Environment Creation and Observation Dimensions
try
    wind_cfg.mode = 'none';
    env = AircraftLandingEnv(P, wind_cfg, false);
    
    obs = env.reset();
    assert(length(obs) == 11, 'Observation vector dimension is not 11');
    assert(~any(isnan(obs)), 'Observation vector contains NaN values');
    assert(~any(isinf(obs)), 'Observation vector contains Inf values');
    
    fprintf('  [PASS] Test 1: Environment creation & observation dimensions verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 1: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 2: Domain Randomization Reset
try
    env_rand = AircraftLandingEnv(P, wind_cfg, true);
    obs1 = env_rand.reset();
    obs2 = env_rand.reset();
    
    % Verify that successive randomized resets generate different initial states
    assert(any(obs1(1:5) ~= obs2(1:5)), 'Domain randomized resets produced identical states');
    
    fprintf('  [PASS] Test 2: Domain randomization reset verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 2: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 3: Environment Step and Reward Function Calculation
try
    test_action = [0.1; -0.05]; % Small control adjustment in [-1, +1]
    [next_obs, reward, is_done, info] = env.step(test_action);
    
    assert(length(next_obs) == 11, 'Next observation dimension is not 11');
    assert(~isnan(reward), 'Reward returned NaN');
    assert(~isinf(reward), 'Reward returned Inf');
    assert(isfield(info, 'reward_breakdown'), 'Info missing reward breakdown');
    assert(isfield(info, 'status'), 'Info missing landing status');
    
    fprintf('  [PASS] Test 3: Environment step and reward calculation verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 3: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 4: Standalone RL Engine Inference and Buffer
try
    cfg = rlConfig();
    agent = rlEngine('create_td3', cfg);
    
    action = rlEngine('getAction', agent, obs, false);
    assert(length(action) == 2, 'Action vector dimension is not 2');
    assert(all(action >= -1.0 & action <= 1.0), 'Action outside [-1, +1] bounds');
    
    agent = rlEngine('storeTransition', agent, obs, action, reward, next_obs, is_done);
    assert(agent.buffer.size == 1, 'Replay buffer transition storage failed');
    
    fprintf('  [PASS] Test 4: Standalone neural policy inference & buffer verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 4: %s\n', ME.message);
    results.failed = results.failed + 1;
end

end
