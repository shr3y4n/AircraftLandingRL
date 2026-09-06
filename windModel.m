function [wind_vel, wind_dot] = windModel(t, x_state, wind_cfg)
% WINDMODEL Computes atmospheric wind velocities and time derivatives.
%
% Accurately separates air-relative and ground-relative velocity vectors.
% Supports steady winds, boundary layer wind shear, discrete 1-cosine gusts,
% and low-altitude Dryden stochastic turbulence filters.
%
% Inputs:
%   t        - Current simulation time [s]
%   x_state  - Current state vector [x; h; V; gamma; theta; q] or struct
%   wind_cfg - Struct defining wind disturbance configuration:
%              .mode        - 'none', 'steady', 'shear', 'gust', 'turbulence'
%              .V_headwind  - Base steady headwind (>0 is headwind, i.e. wx < 0) [m/s]
%              .w_updraft   - Base steady vertical updraft [m/s]
%              .gust_Ax     - Horizontal 1-cosine gust amplitude [m/s]
%              .gust_Ah     - Vertical 1-cosine gust amplitude [m/s]
%              .gust_start  - Gust start time [s]
%              .gust_dur    - Gust duration [s]
%              .turb_level  - Turbulence intensity: 'light', 'moderate', 'severe'
%              .seed        - Random seed for reproducible turbulence
%
% Outputs:
%   wind_vel - [wx; wh] Atmospheric wind velocity in inertial frame [m/s]
%              wx: longitudinal wind along +x (positive tailwind, negative headwind)
%              wh: vertical wind along +h (positive updraft, negative downdraft)
%   wind_dot - [dwx_dt; dwh_dt] Wind acceleration components [m/s^2]

if nargin < 3 || isempty(wind_cfg)
    wind_cfg.mode = 'none';
end

if isstruct(x_state)
    h = max(0.1, x_state.h);
else
    h = max(0.1, x_state(2));
end

wx     = 0.0;
wh     = 0.0;
dwx_dt = 0.0;
dwh_dt = 0.0;

mode = lower(wind_cfg.mode);

switch mode
    case 'none'
        % Zero wind condition
        wx = 0.0;
        wh = 0.0;

    case 'steady'
        % Constant horizontal headwind/tailwind and vertical current
        V_hw = 0.0;
        if isfield(wind_cfg, 'V_headwind')
            V_hw = wind_cfg.V_headwind;
        end
        w_up = 0.0;
        if isfield(wind_cfg, 'w_updraft')
            w_up = wind_cfg.w_updraft;
        end
        wx = -V_hw;  % Headwind opposes aircraft forward motion (+x)
        wh = w_up;

    case 'shear'
        % Boundary layer logarithmic wind shear profile:
        % wx(h) = -V_ref * log(h / z0 + 1) / log(href / z0 + 1)
        V_ref = 8.0;
        if isfield(wind_cfg, 'V_headwind'), V_ref = wind_cfg.V_headwind; end
        href  = 100.0; % Reference altitude [m]
        z0    = 0.05;  % Aerodynamic surface roughness length [m] (runway flat terrain)
        
        profile = log(h / z0 + 1.0) / log(href / z0 + 1.0);
        profile = min(1.5, max(0.05, profile));
        wx = -V_ref * profile;
        wh = 0.0;

    case 'gust'
        % MIL-F-8785C / FAA Standard 1-Cosine Discrete Gust:
        % w(t) = (A/2) * (1 - cos(2*pi*(t - t_start) / d_gust))
        V_hw = 5.0;
        if isfield(wind_cfg, 'V_headwind'), V_hw = wind_cfg.V_headwind; end
        Ax = 6.0;   % Horizontal gust amplitude [m/s]
        Ah = -2.5;  % Vertical gust amplitude [m/s] (downdraft)
        t0 = 15.0;  % Gust start time [s]
        T_g = 6.0;  % Gust duration [s]
        
        if isfield(wind_cfg, 'gust_Ax'),    Ax = wind_cfg.gust_Ax; end
        if isfield(wind_cfg, 'gust_Ah'),    Ah = wind_cfg.gust_Ah; end
        if isfield(wind_cfg, 'gust_start'), t0 = wind_cfg.gust_start; end
        if isfield(wind_cfg, 'gust_dur'),   T_g = wind_cfg.gust_dur; end
        
        wx = -V_hw;
        wh = 0.0;
        
        if (t >= t0) && (t <= t0 + T_g)
            phase = 2.0 * pi * (t - t0) / T_g;
            gust_x = 0.5 * Ax * (1.0 - cos(phase));
            gust_h = 0.5 * Ah * (1.0 - cos(phase));
            
            d_gust_x = 0.5 * Ax * (2.0 * pi / T_g) * sin(phase);
            d_gust_h = 0.5 * Ah * (2.0 * pi / T_g) * sin(phase);
            
            wx = wx - gust_x;
            wh = wh + gust_h;
            dwx_dt = -d_gust_x;
            dwh_dt = d_gust_h;
        end

    case 'turbulence'
        % Dryden Low-Altitude Continuous Stochastic Turbulence Model (MIL-F-8785C)
        % Superimposed on steady headwind
        V_hw = 6.0;
        if isfield(wind_cfg, 'V_headwind'), V_hw = wind_cfg.V_headwind; end
        
        % Turbulence intensity at 20 ft (6 m):
        % light: W20 = 15 kt (~7.7 m/s); moderate: 30 kt (~15.4 m/s); severe: 45 kt (~23.1 m/s)
        W20 = 7.7; % Light turbulence default
        if isfield(wind_cfg, 'turb_level')
            switch lower(wind_cfg.turb_level)
                case 'light',    W20 = 7.7;
                case 'moderate', W20 = 15.4;
                case 'severe',   W20 = 23.1;
            end
        end
        
        % Low-altitude Dryden parameters as function of altitude h [m]
        h_ft = max(10.0, h * 3.28084);
        sigma_w = 0.1 * W20; % [m/s]
        sigma_u = sigma_w / (0.177 + 0.000823 * h_ft)^0.4; % [m/s]
        
        % Multi-frequency harmonic representation of Dryden spectrum
        % Replicable pseudo-random phases
        seed = 42;
        if isfield(wind_cfg, 'seed'), seed = wind_cfg.seed; end
        
        % 10 harmonic frequencies for longitudinal and vertical turbulence
        omega_k = [0.10, 0.25, 0.45, 0.70, 1.10, 1.65, 2.40, 3.50, 5.00, 7.50];
        phase_u = sin(seed * (1:10) * 1.618);
        phase_w = cos(seed * (1:10) * 2.718);
        
        turb_u = 0;
        turb_w = 0;
        dturb_u = 0;
        dturb_w = 0;
        
        for k = 1:length(omega_k)
            amp_k = 1.0 / sqrt(1.0 + (omega_k(k) * 1.5)^2);
            turb_u  = turb_u  + amp_k * sin(omega_k(k) * t + phase_u(k));
            turb_w  = turb_w  + amp_k * sin(omega_k(k) * t + phase_w(k));
            dturb_u = dturb_u + amp_k * omega_k(k) * cos(omega_k(k) * t + phase_u(k));
            dturb_w = dturb_w + amp_k * omega_k(k) * cos(omega_k(k) * t + phase_w(k));
        end
        
        % Normalize and scale
        norm_factor = sqrt(sum(1.0 ./ (1.0 + (omega_k * 1.5).^2)));
        turb_u = (turb_u / norm_factor) * sigma_u;
        turb_w = (turb_w / norm_factor) * sigma_w;
        dturb_u = (dturb_u / norm_factor) * sigma_u;
        dturb_w = (dturb_w / norm_factor) * sigma_w;
        
        wx = -V_hw + turb_u;
        wh = turb_w;
        dwx_dt = dturb_u;
        dwh_dt = dturb_w;

    otherwise
        warning('windModel:UnknownMode', 'Unknown wind mode %s. Defaulting to none.', wind_cfg.mode);
        wx = 0.0;
        wh = 0.0;
end

wind_vel = [wx; wh];
wind_dot = [dwx_dt; dwh_dt];

end
