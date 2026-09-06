function [env, obs_info, act_info] = createEnvironment(P, wind_cfg, randomize_init)
% CREATEENVIRONMENT Factory function for the aircraft landing RL environment.
%
% Checks for the presence of the MATLAB Reinforcement Learning Toolbox.
% If available, constructs an official rlFunctionEnv wrapper with numeric specs.
% If not available, returns the standalone AircraftLandingEnv instance.
%
% Inputs:
%   P              - Aircraft parameters struct from aircraftParameters()
%   wind_cfg       - Wind configuration struct (optional)
%   randomize_init - Boolean flag to randomize initial conditions (optional)
%
% Outputs:
%   env      - Environment object (rlFunctionEnv or AircraftLandingEnv)
%   obs_info - Observation specification
%   act_info - Action specification

if nargin < 1 || isempty(P)
    P = aircraftParameters();
end
if nargin < 2 || isempty(wind_cfg)
    wind_cfg.mode = 'none';
end
if nargin < 3 || isempty(randomize_init)
    randomize_init = false;
end

base_env = AircraftLandingEnv(P, wind_cfg, randomize_init);
obs_info = base_env.ObservationInfo;
act_info = base_env.ActionInfo;

% Check if official RL Toolbox function environment builder exists
if exist('rlFunctionEnv', 'file') == 2 || exist('rlFunctionEnv', 'class') == 8
    try
        step_fcn  = @(action, info) stepWrapper(base_env, action);
        reset_fcn = @() resetWrapper(base_env);
        
        env = rlFunctionEnv(obs_info, act_info, step_fcn, reset_fcn);
        fprintf('[RL Environment] Successfully created official MATLAB rlFunctionEnv.\n');
        return;
    catch ME
        fprintf('[RL Environment] rlFunctionEnv fallback to AircraftLandingEnv: %s\n', ME.message);
    end
end

% Standalone environment
env = base_env;
fprintf('[RL Environment] Created standalone AircraftLandingEnv environment.\n');

end

%% Local Wrapper Functions for MATLAB RL Toolbox
function [next_obs, reward, is_done, info] = stepWrapper(base_env, action)
    [next_obs, reward, is_done, info] = base_env.step(action);
end

function obs = resetWrapper(base_env)
    obs = base_env.reset();
end
