function summary = runMonteCarlo(num_trials, agent_or_path, P)
% RUNMONTECARLO Executes automated Monte Carlo evaluation batteries for PID and RL controllers.
%
% Tests 8 rigorous benchmark experimental suites:
%   1. PID / No Disturbance
%   2. RL  / No Disturbance
%   3. PID / Steady Wind (Headwind & Cross-shear)
%   4. RL  / Steady Wind
%   5. PID / Gusts & Turbulence (1-cosine & Dryden)
%   6. RL  / Gusts & Turbulence
%   7. PID / Randomized Initial Conditions (Offsets in altitude, speed, position)
%   8. RL  / Randomized Initial Conditions
%
% Inputs:
%   num_trials    - Number of trials per suite (e.g. 50 or 100)
%   agent_or_path - Loaded agent struct, agent file path, or empty (defaults to PID-only or best agent)
%   P             - Aircraft parameters struct (optional)
%
% Outputs:
%   summary - Struct array summarizing statistical metrics across all 8 suites

if nargin < 1 || isempty(num_trials), num_trials = 50; end
if nargin < 3 || isempty(P),          P = aircraftParameters(); end

% Ensure results directories exist
res_dir  = fullfile(pwd, 'Results', 'Data');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

fprintf('\n========================================================================\n');
fprintf('  STARTING MONTE CARLO LANDING EVALUATION (%d TRIALS PER BATTERY)\n', num_trials);
fprintf('========================================================================\n');

%% 1. Locate RL Agent if available
agent = [];
if nargin >= 2 && ~isempty(agent_or_path)
    if ischar(agent_or_path) || isstring(agent_or_path)
        if exist(agent_or_path, 'file')
            agent = rlEngine('load', agent_or_path);
        end
    else
        agent = agent_or_path;
    end
else
    % Search default checkpoint directory
    chk_files = dir(fullfile(res_dir, 'checkpoints', '*.mat'));
    if ~isempty(chk_files)
        latest_file = fullfile(chk_files(end).folder, chk_files(end).name);
        try
            agent = rlEngine('load', latest_file);
            fprintf('[Monte Carlo] Loaded available RL checkpoint: %s\n', latest_file);
        catch
            agent = [];
        end
    end
end

has_rl = ~isempty(agent);
if ~has_rl
    fprintf('[Monte Carlo] Notice: No trained RL agent provided or found.\n');
    fprintf('[Monte Carlo] RL benchmark suites will be skipped until training is performed.\n');
end

%% 2. Define Experimental Test Batteries
suites = {
    'PID / No Wind',         'pid', 'none',       false;
    'PID / Steady Wind',     'pid', 'steady',     false;
    'PID / Gusts & Turb',    'pid', 'gust_turb',  false;
    'PID / Randomized IC',   'pid', 'none',       true;
};

if has_rl
    rl_suites = {
        'RL / No Wind',          'rl',  'none',       false;
        'RL / Steady Wind',      'rl',  'steady',     false;
        'RL / Gusts & Turb',     'rl',  'gust_turb',  false;
        'RL / Randomized IC',    'rl',  'none',       true;
    };
    % Interleave or append
    all_suites = [suites; rl_suites];
else
    all_suites = suites;
end

num_suites = size(all_suites, 1);
summary = struct();

%% 3. Execute Monte Carlo Trials
for s_idx = 1:num_suites
    suite_name = all_suites{s_idx, 1};
    ctrl_type  = all_suites{s_idx, 2};
    wind_type  = all_suites{s_idx, 3};
    is_random  = all_suites{s_idx, 4};
    
    fprintf('\n--- Running Suite [%d/%d]: %-30s ---\n', s_idx, num_suites, suite_name);
    
    success_count = 0;
    td_pos_err    = zeros(num_trials, 1);
    td_spd_err    = zeros(num_trials, 1);
    td_sink_rate  = zeros(num_trials, 1);
    td_pitch      = zeros(num_trials, 1);
    rms_eh        = zeros(num_trials, 1);
    rms_eV        = zeros(num_trials, 1);
    ctrl_effort   = zeros(num_trials, 1);
    
    % Fixed master seed per suite for rigorous reproducibility
    rng(2000 + s_idx * 100, 'twister');
    
    for tr = 1:num_trials
        % Configure Disturbance
        wind_cfg = struct();
        switch wind_type
            case 'none'
                wind_cfg.mode = 'none';
            case 'steady'
                wind_cfg.mode = 'steady';
                % Distribute headwind between 2 and 10 m/s
                wind_cfg.V_headwind = 2.0 + rand() * 8.0;
                wind_cfg.w_updraft  = (rand() * 2.0 - 1.0) * 0.8;
            case 'gust_turb'
                wind_cfg.mode = 'turbulence';
                wind_cfg.V_headwind = 4.0 + rand() * 4.0;
                wind_cfg.turb_level = 'moderate';
                wind_cfg.seed = randi(10000);
        end
        
        % Configure Initial Condition
        custom_init = struct();
        if is_random
            custom_init.h0     = P.h0 + (rand() * 2.0 - 1.0) * 20.0;
            custom_init.V0     = P.V_app + (rand() * 2.0 - 1.0) * 4.0;
            custom_init.x0     = P.x0 + (rand() * 2.0 - 1.0) * 150.0;
            custom_init.theta0 = P.gamma_des + deg2rad(4.2 + (rand()*2.0 - 1.0)*1.5);
        end
        
        % Run Trial
        if strcmp(ctrl_type, 'pid')
            [~, m] = simulatePID(P, wind_cfg, custom_init);
        else
            [m, ~] = evaluateAgent(agent, P, wind_cfg, custom_init);
        end
        
        success_count = success_count + double(m.is_success);
        td_pos_err(tr)   = m.touchdown_pos_error;
        td_spd_err(tr)   = m.touchdown_speed_error;
        td_sink_rate(tr) = m.touchdown_sink;
        td_pitch(tr)     = rad2deg(m.touchdown_pitch);
        rms_eh(tr)       = m.rms_glideslope_error;
        rms_eV(tr)       = m.rms_airspeed_error;
        ctrl_effort(tr)  = m.total_control_effort;
    end
    
    % Store Statistics
    summary(s_idx).name            = suite_name;
    summary(s_idx).controller      = ctrl_type;
    summary(s_idx).wind_mode       = wind_type;
    summary(s_idx).trials          = num_trials;
    summary(s_idx).success_rate    = (success_count / num_trials) * 100.0;
    summary(s_idx).mean_pos_err    = mean(td_pos_err);
    summary(s_idx).std_pos_err     = std(td_pos_err);
    summary(s_idx).mean_spd_err    = mean(td_spd_err);
    summary(s_idx).std_spd_err     = std(td_spd_err);
    summary(s_idx).mean_sink_rate  = mean(td_sink_rate);
    summary(s_idx).std_sink_rate   = std(td_sink_rate);
    summary(s_idx).mean_pitch      = mean(td_pitch);
    summary(s_idx).std_pitch       = std(td_pitch);
    summary(s_idx).mean_rms_eh     = mean(rms_eh);
    summary(s_idx).std_rms_eh      = std(rms_eh);
    summary(s_idx).mean_rms_eV     = mean(rms_eV);
    summary(s_idx).std_rms_eV      = std(rms_eV);
    summary(s_idx).mean_effort     = mean(ctrl_effort);
    summary(s_idx).std_effort      = std(ctrl_effort);
    
    % Raw Trial Arrays for Histogram Plotting
    summary(s_idx).raw_pos_err     = td_pos_err;
    summary(s_idx).raw_sink_rate   = td_sink_rate;
    summary(s_idx).raw_rms_eh      = rms_eh;
    
    fprintf('  Success Rate:        %5.1f %%\n', summary(s_idx).success_rate);
    fprintf('  Touchdown Pos Error: %6.2f +/- %5.2f m\n', summary(s_idx).mean_pos_err, summary(s_idx).std_pos_err);
    fprintf('  Touchdown Sink Rate: %6.2f +/- %5.2f m/s\n', summary(s_idx).mean_sink_rate, summary(s_idx).std_sink_rate);
    fprintf('  Touchdown Speed Err: %6.2f +/- %5.2f m/s\n', summary(s_idx).mean_spd_err, summary(s_idx).std_spd_err);
    fprintf('  RMS Glide Error:     %6.2f +/- %5.2f m\n', summary(s_idx).mean_rms_eh, summary(s_idx).std_rms_eh);
    fprintf('  Control Effort:      %6.2f +/- %5.2f\n', summary(s_idx).mean_effort, summary(s_idx).std_effort);
end

%% 4. Save Structured Results and CSV Table
mat_file = fullfile(res_dir, 'monte_carlo_results.mat');
save(mat_file, 'summary', 'P', 'num_trials');
fprintf('\n[Monte Carlo] Full results saved to: %s\n', mat_file);

% Export CSV Summary Table
csv_file = fullfile(res_dir, 'monte_carlo_summary.csv');
fid = fopen(csv_file, 'w');
if fid > 0
    fprintf(fid, 'Suite,Controller,Wind,Trials,SuccessRate_pct,MeanPosErr_m,StdPosErr_m,MeanSinkRate_mps,StdSinkRate_mps,MeanSpeedErr_mps,StdSpeedErr_mps,MeanRMSGlide_m,StdRMSGlide_m,MeanEffort\n');
    for s = 1:num_suites
        fprintf(fid, '%s,%s,%s,%d,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n', ...
            summary(s).name, summary(s).controller, summary(s).wind_mode, summary(s).trials, ...
            summary(s).success_rate, summary(s).mean_pos_err, summary(s).std_pos_err, ...
            summary(s).mean_sink_rate, summary(s).std_sink_rate, ...
            summary(s).mean_spd_err, summary(s).std_spd_err, ...
            summary(s).mean_rms_eh, summary(s).std_rms_eh, summary(s).mean_effort);
    end
    fclose(fid);
    fprintf('[Monte Carlo] Summary CSV exported to: %s\n', csv_file);
end

fprintf('========================================================================\n\n');

end
