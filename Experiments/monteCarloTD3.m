function summary = monteCarloTD3(num_trials, checkpoint_path)
% MONTECARLOTD3 Runs a multi-battery Monte Carlo robustness evaluation for the TD3 policy.
%
% Tests the trained TD3 policy across 4 rigorous flight batteries:
%   1. Calm Air (zero disturbances)
%   2. Steady Wind & Boundary Layer Shear (2-10 m/s headwind + updrafts)
%   3. Gusts & Dryden Atmospheric Turbulence (moderate turbulence + random gusts)
%   4. Randomized Initial Conditions (offsets in h0, V0, x0, and pitch)
%
% Inputs:
%   num_trials      - Number of Monte Carlo flights per battery (default: 50)
%   checkpoint_path - (Optional) Path to specific TD3 checkpoint .mat file
%
% Outputs:
%   summary - Struct array containing statistical performance across batteries
%
% Artifacts saved to:
%   Results/Data/monte_carlo_td3.mat
%   Results/Figures/mc_td3_dispersion.png

if nargin < 1 || isempty(num_trials), num_trials = 50; end

%% 1. Path & Directory Setup
exp_dir   = fileparts(mfilename('fullpath'));
repo_root = fileparts(exp_dir);
addpath(genpath(repo_root));

data_dir = fullfile(repo_root, 'Results', 'Data');
chk_dir  = fullfile(data_dir, 'checkpoints');
fig_dir  = fullfile(repo_root, 'Results', 'Figures');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(fig_dir, 'dir'),  mkdir(fig_dir);  end

fprintf('========================================================================\n');
fprintf('  EXPERIMENT: TD3 POLICY MONTE CARLO ROBUSTNESS (%d TRIALS / BATTERY)\n', num_trials);
fprintf('========================================================================\n');

P = aircraftParameters();

%% 2. Locate and Load Agent Checkpoint
if nargin < 2 || isempty(checkpoint_path)
    target_chk = fullfile(chk_dir, 'agent_td3_final.mat');
    if ~exist(target_chk, 'file')
        chk_files = dir(fullfile(chk_dir, 'agent_td3_ep*.mat'));
        if isempty(chk_files)
            error('monteCarloTD3:NoCheckpoint', ...
                  ['No trained TD3 checkpoint found in:\n  %s\n' ...
                   'Please execute trainTD3("train") before running Monte Carlo.'], chk_dir);
        else
            [~, idx] = max([chk_files.datenum]);
            target_chk = fullfile(chk_files(idx).folder, chk_files(idx).name);
        end
    end
else
    target_chk = checkpoint_path;
    if ~exist(target_chk, 'file')
        error('monteCarloTD3:FileNotFound', 'Specified checkpoint does not exist: %s', target_chk);
    end
end

fprintf('Loading TD3 policy from:\n  %s\n\n', target_chk);
agent = rlEngine('load', target_chk);

%% 3. Define Evaluation Batteries
suites = {
    'TD3 / Calm Air',              'none',       false;
    'TD3 / Steady Wind & Shear',   'steady',     false;
    'TD3 / Gusts & Turbulence',    'gust_turb',  false;
    'TD3 / Randomized Initial IC', 'none',       true;
};

num_suites = size(suites, 1);
summary = struct();

for s_idx = 1:num_suites
    suite_name = suites{s_idx, 1};
    wind_type  = suites{s_idx, 2};
    is_random  = suites{s_idx, 3};
    
    fprintf('\n--- Running Suite [%d/%d]: %-32s ---\n', s_idx, num_suites, suite_name);
    
    success_count = 0;
    td_pos_err    = zeros(num_trials, 1);
    td_spd_err    = zeros(num_trials, 1);
    td_sink_rate  = zeros(num_trials, 1);
    td_pitch      = zeros(num_trials, 1);
    rms_eh        = zeros(num_trials, 1);
    rms_eV        = zeros(num_trials, 1);
    ctrl_effort   = zeros(num_trials, 1);
    
    % Fixed seed per suite for rigorous reproducibility
    rng(2000 + s_idx * 100, 'twister');
    
    for tr = 1:num_trials
        % Disturbance Configuration
        wind_cfg = struct();
        switch wind_type
            case 'none'
                wind_cfg.mode = 'none';
            case 'steady'
                wind_cfg.mode = 'steady';
                wind_cfg.V_headwind = 2.0 + rand() * 8.0;
                wind_cfg.w_updraft  = (rand() * 2.0 - 1.0) * 0.8;
            case 'gust_turb'
                wind_cfg.mode = 'turbulence';
                wind_cfg.V_headwind = 4.0 + rand() * 4.0;
                wind_cfg.turb_level = 'moderate';
                wind_cfg.seed = randi(10000);
        end
        
        % Initial Condition Dispersions
        custom_init = struct();
        if is_random
            custom_init.h0     = P.h0 + (rand() * 2.0 - 1.0) * 20.0;
            custom_init.V0     = P.V_app + (rand() * 2.0 - 1.0) * 4.0;
            custom_init.x0     = P.x0 + (rand() * 2.0 - 1.0) * 150.0;
            custom_init.theta0 = P.gamma_des + deg2rad(4.2 + (rand()*2.0 - 1.0)*1.5);
        end
        
        [m, ~] = evaluateAgent(agent, P, wind_cfg, custom_init);
        
        success_count    = success_count + double(m.is_success);
        td_pos_err(tr)   = m.touchdown_pos_error;
        td_spd_err(tr)   = m.touchdown_speed_error;
        td_sink_rate(tr) = m.touchdown_sink;
        td_pitch(tr)     = rad2deg(m.touchdown_pitch);
        rms_eh(tr)       = m.rms_glideslope_error;
        rms_eV(tr)       = m.rms_airspeed_error;
        ctrl_effort(tr)  = m.total_control_effort;
    end
    
    % Store Summary Statistics
    summary(s_idx).name           = suite_name;
    summary(s_idx).controller     = 'TD3';
    summary(s_idx).wind_mode      = wind_type;
    summary(s_idx).trials         = num_trials;
    summary(s_idx).success_rate   = (success_count / num_trials) * 100.0;
    summary(s_idx).mean_pos_err   = mean(td_pos_err);
    summary(s_idx).std_pos_err    = std(td_pos_err);
    summary(s_idx).mean_spd_err   = mean(td_spd_err);
    summary(s_idx).std_spd_err    = std(td_spd_err);
    summary(s_idx).mean_sink_rate = mean(td_sink_rate);
    summary(s_idx).std_sink_rate  = std(td_sink_rate);
    summary(s_idx).mean_pitch     = mean(td_pitch);
    summary(s_idx).std_pitch      = std(td_pitch);
    summary(s_idx).mean_rms_eh    = mean(rms_eh);
    summary(s_idx).std_rms_eh     = std(rms_eh);
    summary(s_idx).mean_rms_eV    = mean(rms_eV);
    summary(s_idx).std_rms_eV     = std(rms_eV);
    summary(s_idx).mean_effort    = mean(ctrl_effort);
    summary(s_idx).std_effort     = std(ctrl_effort);
    
    summary(s_idx).raw_pos_err    = td_pos_err;
    summary(s_idx).raw_sink_rate  = td_sink_rate;
    summary(s_idx).raw_rms_eh     = rms_eh;
    
    fprintf('  Success Rate:        %5.1f %%\n', summary(s_idx).success_rate);
    fprintf('  Touchdown Pos Error: %6.2f +/- %5.2f m\n', summary(s_idx).mean_pos_err, summary(s_idx).std_pos_err);
    fprintf('  Touchdown Sink Rate: %6.2f +/- %5.2f m/s\n', summary(s_idx).mean_sink_rate, summary(s_idx).std_sink_rate);
    fprintf('  RMS Glide Error:     %6.2f +/- %5.2f m\n', summary(s_idx).mean_rms_eh, summary(s_idx).std_rms_eh);
    fprintf('  Control Effort:      %6.2f +/- %5.2f\n', summary(s_idx).mean_effort, summary(s_idx).std_effort);
end

%% 4. Plot & Save Dispersion Figures
fprintf('\nGenerating Monte Carlo dispersion plots...\n');
plotResults('monte_carlo', summary, 'mc_td3_dispersion.png');

%% 5. Save Results
mat_file = fullfile(data_dir, 'monte_carlo_td3.mat');
save(mat_file, 'summary', 'P', 'num_trials');
fprintf('Saved TD3 Monte Carlo dataset to:\n  %s\n', mat_file);
fprintf('========================================================================\n\n');

end
