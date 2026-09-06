function results = testAircraftDynamics()
% TESTAIRCRAFTDYNAMICS Unit tests for aircraft physical equations and trim states.

fprintf('\n>>> Running Aircraft Dynamics & Aerodynamics Tests...\n');

P = aircraftParameters();
results.passed = 0;
results.failed = 0;

%% Test 1: Level Flight Trim Equilibrium
try
    level_opt.gamma0 = 0.0;
    level_opt.V0 = 40.0;
    [s_lvl, trim_lvl] = initializeAircraft(P, level_opt);
    x_vec = [s_lvl.x; s_lvl.h; s_lvl.V; s_lvl.gamma; s_lvl.theta; s_lvl.q];
    u_vec = [trim_lvl.delta_e; trim_lvl.delta_T];
    
    [xdot, ~] = aircraftDynamics(0, x_vec, u_vec, P);
    
    % Check equilibrium derivatives: dV/dt, dgamma/dt, dq/dt ~ 0
    assert(abs(xdot(3)) < 1e-2, 'dV/dt in level trim exceeds tolerance');
    assert(abs(xdot(4)) < 1e-3, 'dgamma/dt in level trim exceeds tolerance');
    assert(abs(xdot(6)) < 1e-3, 'dq/dt in level trim exceeds tolerance');
    
    fprintf('  [PASS] Test 1: Level flight trim equilibrium verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 1: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 2: 3-Degree Approach Glide-Slope Trim
try
    [s_app, trim_app] = initializeAircraft(P);
    x_vec = [s_app.x; s_app.h; s_app.V; s_app.gamma; s_app.theta; s_app.q];
    u_vec = [trim_app.delta_e; trim_app.delta_T];
    
    [xdot, ~] = aircraftDynamics(0, x_vec, u_vec, P);
    
    assert(abs(xdot(3)) < 1e-2, 'dV/dt in glide trim exceeds tolerance');
    assert(abs(xdot(4)) < 1e-3, 'dgamma/dt in glide trim exceeds tolerance');
    assert(abs(xdot(6)) < 1e-3, 'dq/dt in glide trim exceeds tolerance');
    
    fprintf('  [PASS] Test 2: 3-degree glide-slope trim verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 2: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 3: Elevator Pitching Moment Directional Authority (Cmdelta_e < 0)
try
    % Positive elevator (trailing edge down) must produce pitch-down moment (qdot < 0)
    u_pitch_down = [trim_app.delta_e + deg2rad(5.0); trim_app.delta_T];
    [xdot_down, ~] = aircraftDynamics(0, x_vec, u_pitch_down, P);
    assert(xdot_down(6) < -0.1, 'Elevator deflection did not produce expected pitch-down acceleration');
    
    % Negative elevator (trailing edge up) must produce pitch-up moment (qdot > 0)
    u_pitch_up = [trim_app.delta_e - deg2rad(5.0); trim_app.delta_T];
    [xdot_up, ~] = aircraftDynamics(0, x_vec, u_pitch_up, P);
    assert(xdot_up(6) > 0.1, 'Elevator deflection did not produce expected pitch-up acceleration');
    
    fprintf('  [PASS] Test 3: Elevator moment control authority & signs verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 3: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 4: Throttle Authority and Forward Acceleration
try
    % Full throttle must produce positive along-path acceleration
    u_full = [trim_app.delta_e; 1.0];
    [xdot_full, ~] = aircraftDynamics(0, x_vec, u_full, P);
    assert(xdot_full(3) > 0.5, 'Full throttle did not produce positive acceleration');
    
    % Idle throttle must produce deceleration
    u_idle = [trim_app.delta_e; 0.0];
    [xdot_idle, ~] = aircraftDynamics(0, x_vec, u_idle, P);
    assert(xdot_idle(3) < -0.25, 'Idle throttle did not produce deceleration');
    
    fprintf('  [PASS] Test 4: Propulsion throttle authority verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 4: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 5: Nonlinear Aerodynamic Stall Transition
try
    % High AoA test state
    s_stall = s_app;
    s_stall.theta = s_app.gamma + deg2rad(22.0); % alpha = 22 deg (well above stall of 14 deg)
    x_stall = [s_stall.x; s_stall.h; s_stall.V; s_stall.gamma; s_stall.theta; s_stall.q];
    
    [~, info_stall] = aircraftDynamics(0, x_stall, u_vec, P);
    [~, info_nom]   = aircraftDynamics(0, x_vec, u_vec, P);
    
    assert(info_stall.is_stalled, 'Stall condition was not flagged at alpha = 22 deg');
    assert(info_stall.CD > info_nom.CD * 2.0, 'Drag did not dramatically increase post-stall');
    
    fprintf('  [PASS] Test 5: Nonlinear stall modeling & separation verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 5: %s\n', ME.message);
    results.failed = results.failed + 1;
end

%% Test 6: Ground Effect Induced Drag Reduction
try
    s_high = s_app; s_high.h = 200.0; % High above ground
    s_low  = s_app; s_low.h  = 2.0;   % In ground effect (h < b = 10m)
    
    [~, info_high] = aircraftDynamics(0, [s_high.x; s_high.h; s_high.V; s_high.gamma; s_high.theta; s_high.q], u_vec, P);
    [~, info_low]  = aircraftDynamics(0, [s_low.x; s_low.h; s_low.V; s_low.gamma; s_low.theta; s_low.q], u_vec, P);
    
    assert(info_low.CD < info_high.CD, 'Ground effect did not reduce total drag near ground');
    
    fprintf('  [PASS] Test 6: Ground effect induced drag reduction verified.\n');
    results.passed = results.passed + 1;
catch ME
    fprintf('  [FAIL] Test 6: %s\n', ME.message);
    results.failed = results.failed + 1;
end

end
