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

total_passed = 0;
total_failed = 0;
t_start = tic;

% 1. Aircraft Dynamics
res_dyn = testAircraftDynamics();
total_passed = total_passed + res_dyn.passed;
total_failed = total_failed + res_dyn.failed;

% 2. Classical PID Baseline
res_pid = testPID();
total_passed = total_passed + res_pid.passed;
total_failed = total_failed + res_pid.failed;

% 3. RL Environment & Actions
res_env = testEnvironment();
total_passed = total_passed + res_env.passed;
total_failed = total_failed + res_env.failed;

% 4. Landing Criteria Logic
res_land = testLandingLogic();
total_passed = total_passed + res_land.passed;
total_failed = total_failed + res_land.failed;

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
