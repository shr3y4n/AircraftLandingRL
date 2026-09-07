function all_passed = runAllTests()
% RUNALLTESTS Top-level test runner orchestrating full unit and integration test suites.
%
% Runs:
%   1. testAircraftDynamics - equations of motion, trim, elevator/throttle authority, stall
%   2. testPID              - baseline controller stability, disturbance rejection, landing
%   3. testEnvironment      - RL environment observation specs, domain randomization, step
%   4. testLandingLogic     - touchdown classification, sink rates, runway safety boundaries
%
% Returns:
%   all_passed - Boolean true if all test assertions succeed

clc;
fprintf('========================================================================\n');
fprintf('  AIRCRAFTRL: AUTONOMOUS AIRCRAFT LANDING VERIFICATION TEST HARNESS\n');
fprintf('========================================================================\n');

% Ensure repository root and subdirectories are on MATLAB path regardless of working directory
test_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(test_dir);
addpath(genpath(repo_root));

total_passed = 0;
total_failed = 0;
t_start = tic;

% 1. Aircraft Dynamics
try
    res_dyn = testAircraftDynamics();
    total_passed = total_passed + res_dyn.passed;
    total_failed = total_failed + res_dyn.failed;
catch ME
    fprintf('  [CRITICAL SUITE ERROR] testAircraftDynamics failed: %s\n', ME.message);
    total_failed = total_failed + 1;
end

% 2. Classical PID Baseline
try
    res_pid = testPID();
    total_passed = total_passed + res_pid.passed;
    total_failed = total_failed + res_pid.failed;
catch ME
    fprintf('  [CRITICAL SUITE ERROR] testPID failed: %s\n', ME.message);
    total_failed = total_failed + 1;
end

% 3. RL Environment & Actions
try
    res_env = testEnvironment();
    total_passed = total_passed + res_env.passed;
    total_failed = total_failed + res_env.failed;
catch ME
    fprintf('  [CRITICAL SUITE ERROR] testEnvironment failed: %s\n', ME.message);
    total_failed = total_failed + 1;
end

% 4. Landing Criteria Logic
try
    res_land = testLandingLogic();
    total_passed = total_passed + res_land.passed;
    total_failed = total_failed + res_land.failed;
catch ME
    fprintf('  [CRITICAL SUITE ERROR] testLandingLogic failed: %s\n', ME.message);
    total_failed = total_failed + 1;
end

t_elapsed = toc(t_start);

fprintf('\n========================================================================\n');
fprintf('  TEST SUMMARY: %d PASSED, %d FAILED (Elapsed: %.2f s)\n', ...
        total_passed, total_failed, t_elapsed);
if total_failed == 0
    fprintf('  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!\n');
    all_passed = true;
else
    fprintf('  WARNING: SOME TESTS FAILED. PLEASE INSPECT LOGS ABOVE.\n');
    all_passed = false;
end
fprintf('========================================================================\n\n');

end
