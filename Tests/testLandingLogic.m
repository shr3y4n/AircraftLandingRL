function results = testLandingLogic()
% TESTLANDINGLOGIC Unit tests for touchdown classification and flight corridor checks.

fprintf('\n>>> Running Landing Logic & Touchdown Status Tests...\n');

P = aircraftParameters();
results.passed = 0;
results.failed = 0;

%% Test 1: Soft Touchdown Classification
try
    % Altitude 0, sink rate 0.6 m/s, pitch 3 deg, on runway
    s_soft = [350.0; 0.0; 27.5; deg2rad(-1.25); deg2rad(3.0); 0.0];
    [status, is_term, details] = landingScenario('evaluate', s_soft, P);
    
    assert(is_term, 'Touchdown at h=0 was not flagged as terminal');
    assert(strcmp(status, 'SOFT_TOUCHDOWN'), sprintf('Expected SOFT_TOUCHDOWN, got %s', status));
    assert(details.is_success, 'Soft touchdown was not flagged as successful');
    
    fprintf('  [PASS] Test 1: Soft touchdown logic verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 1: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 2: Hard Landing Classification
try
    % Altitude 0, sink rate 2.4 m/s (between 1.8 and 3.5 m/s)
    s_hard = [350.0; 0.0; 28.0; deg2rad(-4.9); deg2rad(3.0); 0.0];
    [status, is_term, details] = landingScenario('evaluate', s_hard, P);
    
    assert(is_term, 'Hard touchdown was not flagged as terminal');
    assert(strcmp(status, 'HARD_LANDING'), sprintf('Expected HARD_LANDING, got %s', status));
    assert(~details.is_success, 'Hard landing incorrectly marked as success');
    
    fprintf('  [PASS] Test 2: Hard landing classification verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 2: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 3: Structural Crash High Sink Rate Classification
try
    % Altitude 0, sink rate 4.2 m/s (> 3.5 m/s)
    s_crash = [350.0; 0.0; 28.0; deg2rad(-8.6); deg2rad(2.0); 0.0];
    [status, is_term, ~] = landingScenario('evaluate', s_crash, P);
    
    assert(is_term, 'High sink rate touchdown was not terminal');
    assert(strcmp(status, 'CRASH_HIGH_SINK_RATE'), sprintf('Expected CRASH_HIGH_SINK_RATE, got %s', status));
    
    fprintf('  [PASS] Test 3: High sink rate crash classification verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 3: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 4: Short-of-Runway Crash Classification
try
    % Altitude 0 before runway threshold (x = -50 m)
    s_short = [-50.0; 0.0; 30.0; deg2rad(-2.0); deg2rad(3.0); 0.0];
    [status, is_term, ~] = landingScenario('evaluate', s_short, P);
    
    assert(is_term, 'Touchdown short of runway was not terminal');
    assert(strcmp(status, 'CRASH_SHORT_OF_RUNWAY'), sprintf('Expected CRASH_SHORT_OF_RUNWAY, got %s', status));
    
    fprintf('  [PASS] Test 4: Short-of-runway crash detection verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 4: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 5: Nose Gear Collapse & Tailstrike Pitch Limits
try
    % Touchdown with nose pitched down (theta < 0.5 deg)
    s_nose = [300.0; 0.0; 28.0; deg2rad(-1.0); deg2rad(-1.0); 0.0];
    [status_nose, ~, ~] = landingScenario('evaluate', s_nose, P);
    assert(strcmp(status_nose, 'NOSE_GEAR_COLLAPSE'), 'Nose-down touchdown not flagged as nose gear collapse');
    
    % Touchdown with tail strike (theta > 8.5 deg)
    s_tail = [300.0; 0.0; 28.0; deg2rad(-1.0); deg2rad(10.0); 0.0];
    [status_tail, ~, ~] = landingScenario('evaluate', s_tail, P);
    assert(strcmp(status_tail, 'TAIL_STRIKE'), 'High pitch touchdown not flagged as tail strike');
    
    fprintf('  [PASS] Test 5: Nose gear collapse and tail strike checks verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 5: %s\n', ME.message);
    results.failed = results.failed + 1;
end

end
