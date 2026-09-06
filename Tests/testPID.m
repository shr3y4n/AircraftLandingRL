function results = testPID()
% TESTPID Validates classical PID flight control system loops and landing performance.

fprintf('\n>>> Running Classical PID Baseline Controller Tests...\n');

P = aircraftParameters();
results.passed = 0;
results.failed = 0;

%% Test 1: Controller Initialization & Reset
try
    [~, trim] = initializeAircraft(P);
    ctrl_state = pidLandingController('reset', P, trim);
    assert(ctrl_state.int_theta == 0.0, 'Pitch integrator not initialized to zero');
    assert(ctrl_state.int_V == 0.0, 'Airspeed integrator not initialized to zero');
    
    fprintf('  [PASS] Test 1: PID controller state reset verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 1: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 2: Nominal Approach and Landing Closed-Loop Simulation
try
    wind_cfg.mode = 'none';
    [flight_log, m] = simulatePID(P, wind_cfg);
    
    assert(m.is_success, sprintf('Nominal PID landing did not succeed. Status: %s', m.status));
    assert(m.touchdown_sink <= P.td_sink_rate_max, 'Touchdown sink rate exceeded safe limit');
    assert(m.touchdown_x >= P.td_x_min && m.touchdown_x <= P.td_x_max, 'Touchdown outside runway zone');
    assert(m.rms_glideslope_error < 5.0, 'RMS glide-slope tracking error exceeds 5.0 m');
    assert(m.rms_airspeed_error < 2.5, 'RMS airspeed tracking error exceeds 2.5 m/s');
    
    fprintf('  [PASS] Test 2: Nominal PID approach and soft landing verified.\n');
    fprintf('         Touchdown: x=%.1fm, hdot=%.2fm/s, V=%.1fm/s, Status=%s\n', ...
            m.touchdown_x, m.touchdown_sink, m.touchdown_V, m.status);
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 2: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 3: Disturbance Rejection under Headwind & Gusts
try
    wind_cfg.mode = 'gust';
    wind_cfg.V_headwind = 6.0;
    wind_cfg.gust_Ax = 5.0;
    wind_cfg.gust_Ah = -2.0;
    wind_cfg.gust_start = 20.0;
    wind_cfg.gust_dur = 6.0;
    
    [flight_log_gust, m_gust] = simulatePID(P, wind_cfg);
    
    assert(m_gust.is_success, sprintf('PID landing under wind gust failed. Status: %s', m_gust.status));
    assert(m_gust.touchdown_sink <= P.td_sink_rate_max, 'Touchdown sink rate in gusts exceeded limit');
    
    fprintf('  [PASS] Test 3: PID disturbance rejection under gusts verified.\n');
    fprintf('         Touchdown in Gusts: x=%.1fm, hdot=%.2fm/s, Status=%s\n', ...
            m_gust.touchdown_x, m_gust.touchdown_sink, m_gust.status);
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 3: %s\n', ME.message);
    results.failed = results.failed + 1;
end

end
