classdef AircraftLandingEnv < handle
% AIRCRAFTLANDINGENV Custom simulation environment for autonomous landing.
%
% Compatible with both MATLAB Reinforcement Learning Toolbox interfaces
% and standalone pure-MATLAB control scripts.
%
% Observation Space (11 continuous states, normalized):
%   1.  x / 3000           - Distance to threshold
%   2.  h / 200            - Altitude above runway
%   3.  e_h / 50           - Altitude error relative to reference
%   4.  e_V / 10           - Airspeed error relative to reference
%   5.  hdot / 10          - Vertical sink rate
%   6.  theta              - Pitch attitude [rad]
%   7.  q                  - Pitch rate [rad/s]
%   8.  gamma              - Flight-path angle [rad]
%   9.  alpha              - Angle of attack [rad]
%   10. prev_delta_e       - Previous elevator command [rad]
%   11. prev_delta_T       - Previous throttle command [-]
%
% Action Space (2 continuous controls normalized in [-1, +1]):
%   a(1): Normalized elevator deflection delta_e
%   a(2): Normalized throttle fraction delta_T
%
% Methods:
%   env = AircraftLandingEnv(P, wind_cfg, randomize_init)
%   obs = env.reset()
%   [next_obs, reward, is_done, info] = env.step(action)

    properties
        P                  % Aircraft parameters struct
        wind_cfg           % Wind disturbance configuration struct
        randomize_init     % Flag to enable domain randomization on reset
        
        state              % Current physical state vector [6x1]
        prev_controls      % Previous control action [delta_e; delta_T]
        trim               % Approach trim struct
        step_count         % Current episode step counter
        max_steps          % Maximum allowed steps per episode
        current_time       % Elapsed simulation time [s]
        
        last_info          % Diagnostic telemetry from previous step
        action_history     % Stored actions for the episode
        state_history      % Stored states for the episode
    end
    
    properties (SetAccess = private)
        ObservationInfo    % Environment observation specification
        ActionInfo         % Environment action specification
    end

    methods
        function this = AircraftLandingEnv(P, wind_cfg, randomize_init)
            % Constructor
            if nargin < 1 || isempty(P)
                this.P = aircraftParameters();
            else
                this.P = P;
            end
            
            if nargin < 2 || isempty(wind_cfg)
                this.wind_cfg.mode = 'none';
            else
                this.wind_cfg = wind_cfg;
            end
            
            if nargin < 3 || isempty(randomize_init)
                this.randomize_init = false;
            else
                this.randomize_init = randomize_init;
            end
            
            [~, this.trim] = initializeAircraft(this.P);
            this.max_steps = round(this.P.t_max / this.P.dt);
            
            % Setup specification structures for RL Toolbox compatibility
            this.setupSpecs();
            
            % Initial reset
            this.reset();
        end

        function setupSpecs(this)
            % Configures RL Toolbox specs if available, otherwise mock structs
            obs_dim = 11;
            act_dim = 2;
            
            if exist('rlNumericSpec', 'file') == 2 || exist('rlNumericSpec', 'class') == 8
                try
                    this.ObservationInfo = rlNumericSpec([obs_dim, 1], ...
                        'LowerLimit', -ones(obs_dim, 1) * 10, ...
                        'UpperLimit', ones(obs_dim, 1) * 10);
                    this.ObservationInfo.Name = 'Observation';
                    this.ObservationInfo.Description = 'Aircraft landing telemetry';
                    
                    this.ActionInfo = rlNumericSpec([act_dim, 1], ...
                        'LowerLimit', [-1.0; -1.0], ...
                        'UpperLimit', [1.0; 1.0]);
                    this.ActionInfo.Name = 'Action';
                    this.ActionInfo.Description = '[Elevator, Throttle] in [-1, +1]';
                catch
                    this.setupFallbackSpecs(obs_dim, act_dim);
                end
            else
                this.setupFallbackSpecs(obs_dim, act_dim);
            end
        end

        function setupFallbackSpecs(this, obs_dim, act_dim)
            this.ObservationInfo.Dimension = [obs_dim, 1];
            this.ObservationInfo.LowerLimit = -ones(obs_dim, 1) * 10;
            this.ObservationInfo.UpperLimit = ones(obs_dim, 1) * 10;
            this.ActionInfo.Dimension = [act_dim, 1];
            this.ActionInfo.LowerLimit = [-1.0; -1.0];
            this.ActionInfo.UpperLimit = [1.0; 1.0];
        end

        function obs = reset(this, custom_init)
            % Resets the environment to initial approach state
            this.step_count   = 0;
            this.current_time = 0.0;
            
            init_opts = struct();
            
            if nargin >= 2 && ~isempty(custom_init)
                init_opts = custom_init;
            elseif this.randomize_init
                % Domain randomization for robust policy training
                % Altitude +/- 15 m
                dh0 = (rand() * 2.0 - 1.0) * 15.0;
                init_opts.h0 = this.P.h0 + dh0;
                
                % Airspeed +/- 3.0 m/s (~6 kt)
                dV0 = (rand() * 2.0 - 1.0) * 3.0;
                init_opts.V0 = this.P.V_app + dV0;
                
                % Distance +/- 100 m
                dx0 = (rand() * 2.0 - 1.0) * 100.0;
                init_opts.x0 = this.P.x0 + dx0;
                
                % Pitch attitude slight perturbation
                dtheta = deg2rad((rand() * 2.0 - 1.0) * 1.5);
                init_opts.theta0 = this.trim.theta + dtheta;
            end
            
            init_s = initializeAircraft(this.P, init_opts);
            this.state = [init_s.x; init_s.h; init_s.V; init_s.gamma; init_s.theta; init_s.q];
            this.prev_controls = [this.trim.delta_e; this.trim.delta_T];
            
            obs = this.getObservation();
            
            this.action_history = [];
            this.state_history  = this.state;
        end

        function [next_obs, reward, is_done, info] = step(this, action)
            % Executes one simulation step of duration dt
            this.step_count   = this.step_count + 1;
            this.current_time = this.current_time + this.P.dt;
            
            % 1. Map Normalized RL Action [-1, +1] to Physical Controls
            controls = this.scaleAction(action);
            
            % 2. Advance Aircraft Dynamics via RK4
            [next_state_vec, step_info] = updateAircraft(this.state, controls, ...
                                                         this.P.dt, this.P, this.wind_cfg);
            
            % 3. Evaluate Landing & Safety Status
            landing_status = landingScenario('evaluate', next_state_vec, this.P);
            
            % 4. Compute Reward
            [reward, reward_breakdown] = rewardFunction(next_state_vec, controls, ...
                                                        this.prev_controls, this.P, landing_status);
            
            % 5. Check Termination Conditions
            is_done = landing_status.is_terminal || (this.step_count >= this.max_steps);
            
            % Update Internal State
            this.state         = next_state_vec;
            this.prev_controls = controls;
            this.last_info     = step_info;
            
            % 6. Construct Next Observation
            next_obs = this.getObservation();
            
            % 7. Diagnostic Package
            info.status           = landing_status.status;
            info.is_success       = landing_status.details.is_success;
            info.reward_breakdown = reward_breakdown;
            info.step_info        = step_info;
            info.controls         = controls;
            info.raw_action       = action;
            info.sim_time         = this.current_time;
            
            this.state_history(:, end+1)  = this.state;
            this.action_history(:, end+1) = controls;
        end

        function obs = getObservation(this)
            % Normalizes physical states into continuous observation vector
            x     = this.state(1);
            h     = this.state(2);
            V     = this.state(3);
            gamma = this.state(4);
            theta = this.state(5);
            q     = this.state(6);
            
            ref = landingScenario('reference', x, h, this.P);
            e_h = ref.h_ref - h;
            e_V = ref.V_ref - V;
            hdot = V * sin(gamma);
            alpha = theta - gamma;
            
            obs = zeros(11, 1);
            obs(1)  = x / this.P.rl.obs_norm.x;
            obs(2)  = h / this.P.rl.obs_norm.h;
            obs(3)  = e_h / this.P.rl.obs_norm.eh;
            obs(4)  = e_V / this.P.rl.obs_norm.eV;
            obs(5)  = hdot / this.P.rl.obs_norm.hdot;
            obs(6)  = theta;
            obs(7)  = q;
            obs(8)  = gamma;
            obs(9)  = alpha;
            obs(10) = this.prev_controls(1) / this.P.elevator_max;
            obs(11) = this.prev_controls(2);
        end

        function controls = scaleAction(this, action)
            % Scales continuous action in [-1, +1] to physical actuator commands
            a1 = min(1.0, max(-1.0, action(1)));
            a2 = min(1.0, max(-1.0, action(2)));
            
            % Elevator: action -1 to +1 maps across [elevator_min, elevator_max]
            % centered about trim elevator
            if a1 >= 0
                delta_e = this.trim.delta_e + a1 * (this.P.elevator_max - this.trim.delta_e);
            else
                delta_e = this.trim.delta_e + a1 * (this.trim.delta_e - this.P.elevator_min);
            end
            
            % Throttle: action -1 to +1 maps across [throttle_min, throttle_max]
            % centered about trim throttle
            if a2 >= 0
                delta_T = this.trim.delta_T + a2 * (this.P.throttle_max - this.trim.delta_T);
            else
                delta_T = this.trim.delta_T + a2 * (this.trim.delta_T - this.P.throttle_min);
            end
            
            delta_e = min(this.P.elevator_max, max(this.P.elevator_min, delta_e));
            delta_T = min(this.P.throttle_max, max(this.P.throttle_min, delta_T));
            
            controls = [delta_e; delta_T];
        end
    end
end
