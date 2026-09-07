# Autonomous Aircraft Landing: Classical Control & Deep Reinforcement Learning

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020a--R2026a-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passed-brightgreen.svg)](Tests/)

A research-grade simulation, guidance, and flight control framework for autonomous fixed-wing aircraft approach and landing. This repository evaluates classical multi-loop gain-scheduled PID control alongside Deep Reinforcement Learning (Twin Delayed DDPG - TD3, and Soft Actor-Critic - SAC) under atmospheric wind shear, stochastic gusts, and initial condition dispersions.

---

## 1. System Architecture & Flight Dynamics

### 1.1 Longitudinal Nonlinear Equations of Motion
The aircraft dynamics are modeled via continuous 6-DOF longitudinal equations of motion formulated in the air-relative wind axes:

$$\dot{x} = V \cos\gamma + w_x$$
$$\dot{h} = V \sin\gamma + w_h$$
$$\dot{V} = \frac{T \cos\alpha - D}{m} - g \sin\gamma - (\dot{w}_x \cos\gamma + \dot{w}_h \sin\gamma)$$
$$\dot{\gamma} = \frac{T \sin\alpha + L}{m V} - \frac{g \cos\gamma}{V} + \frac{\dot{w}_x \sin\gamma - \dot{w}_h \cos\gamma}{V}$$
$$\dot{\theta} = q$$
$$\dot{q} = \frac{M}{I_{yy}}$$

Where:
- $\mathbf{x} = [x, h, V, \gamma, \theta, q]^T$: horizontal position along runway, geometric altitude, true airspeed, air-relative flight-path angle, pitch attitude, and body-axis pitch rate.
- $\alpha = \theta - \gamma$: aerodynamic angle of attack.
- $\mathbf{w} = [w_x, w_h]^T$: horizontal and vertical atmospheric wind components.

### 1.2 Aerodynamic Coefficients with $C^1$ Smoothstep Stall
Aerodynamic forces and pitching moments incorporate induced drag, elevator control power, pitch damping, and high-angle-of-attack separation:

$$L = \frac{1}{2} \rho_0 V^2 S C_L, \quad D = \frac{1}{2} \rho_0 V^2 S C_D, \quad M = \frac{1}{2} \rho_0 V^2 S \bar{c} C_m, \quad T = \delta_T T_{\text{max}}$$

Linear pre-stall coefficients:
$$C_{L,\text{lin}} = C_{L0} + C_{L\alpha}\alpha + C_{L\delta_e}\delta_e + C_{Lq}\left(\frac{\bar{c}}{2V}q\right)$$
$$C_m = C_{m0} + C_{m\alpha}\alpha + C_{m\delta_e}\delta_e + C_{mq}\left(\frac{\bar{c}}{2V}q\right)$$

Nonlinear stall blending: A smoothstep $C^1$ Hermite polynomial transition is applied between $\alpha_{\text{stall}} \pm \Delta\alpha$ ($12^\circ \to 16^\circ$), ensuring strict linear aerodynamics during nominal approach while preventing unbounded lift and enforcing post-stall drag divergence.

### 1.3 Ground Effect (Wieselsberger Model)
Near runway contact ($h < b = 10\text{ m}$), induced drag is reduced via:
$$K_{\text{eff}}(h) = K \cdot \frac{(16 (h/b))^2}{1 + (16 (h/b))^2}$$

---

## 2. Approach & Landing Guidance

```
Altitude (m)
 173m |  Aircraft (V=35 m/s, gamma=-3.0 deg)
      |   \
      |    \   Approach Corridor (3-degree Glide Slope)
      |     \
  12m |......\................ Flare Initiation (h_flare = 12m)
      |       \~~~~~-.._       Exponential flare curve (hdot -> -0.5 m/s)
   0m +--------|--------*--------------------> Distance x (m)
             x=0       x=300m (Touchdown Aim Point)
          Threshold
```

### 2.1 Approach & Flare Reference Profiles
- **Approach Phase ($h > 12\text{ m}$)**: Constant $-3.0^\circ$ glide slope targeting touchdown aim point $x = 300\text{ m}$. Target airspeed $V_{\text{app}} = 35.0\text{ m/s}$ ($68\text{ kt}$).
- **Flare Phase ($h \le 12\text{ m}$)**: Exponential sink-rate reduction smoothly transitioning descent rate from $\dot{h}_{\text{gs}} = -1.83\text{ m/s}$ to touchdown target $\dot{h}_{\text{td}} = -0.50\text{ m/s}$ and bleeding speed toward $V_{\text{td}} = 28.0\text{ m/s}$.

### 2.2 Touchdown Criteria & Safety Limits
- **Soft Touchdown**: $|\dot{h}_{\text{td}}| \le 1.0\text{ m/s}$ within runway zone $x \in [0, 650]\text{ m}$.
- **Acceptable Landing**: $|\dot{h}_{\text{td}}| \le 1.8\text{ m/s}$.
- **Hard Landing**: $1.8 < |\dot{h}_{\text{td}}| \le 3.5\text{ m/s}$ (structural inspection required).
- **Crash**: $|\dot{h}_{\text{td}}| > 3.5\text{ m/s}$, $x < 0\text{ m}$ (short of runway), $x > 650\text{ m}$ (overrun), $\theta < 0.5^\circ$ (nose-gear collapse), or $\theta > 8.5^\circ$ (tailstrike).

---

## 3. Control Architectures

### 3.1 Classical Multi-Loop PID Baseline
- **Inner Loop (Pitch Rate Damping & Attitude Tracking)**:
  $$\delta_e = \delta_{e,\text{trim}} - \left( K_{p,\theta}(\theta_{\text{cmd}} - \theta) + K_{i,\theta}\int(\theta_{\text{cmd}} - \theta)dt - K_{d,\theta} q \right)$$
  Equipped with anti-windup conditional integration, rate limiting ($40^\circ/\text{s}$), and derivative filtering ($N_\theta = 30$).
- **Outer Guidance Loop**: Computes $\theta_{\text{cmd}}$ from altitude tracking error $\Delta h$ during approach, smoothly switching to flare sink-rate regulation below $12\text{ m}$.
- **Autothrottle**: PI airspeed regulation maintaining target speed $V_{\text{ref}}$ with anti-windup clamping.

### 3.2 Deep Reinforcement Learning (TD3 & SAC)
- **Continuous Observation Space (11 states)**:
  $$\mathbf{s} = \left[ \frac{x}{3000}, \frac{h}{200}, \frac{e_h}{50}, \frac{e_V}{10}, \frac{\dot{h}}{10}, \theta, q, \gamma, \alpha, \delta_{e,\text{prev}}, \delta_{T,\text{prev}} \right]^T$$
- **Continuous Action Space (2 controls in $[-1, +1]$)**:
  $$\mathbf{a} = [\Delta\delta_e, \Delta\delta_T]^T \to \text{scaled to physical actuator limits}$$
- **Reward Formulation**:
  $$r = r_{\text{glide}} + r_{\text{airspeed}} + r_{\text{sink}} + r_{\text{pitch}} + r_{\text{alpha}} + r_{\text{effort}} + r_{\text{smoothness}} + r_{\text{progress}} + r_{\text{terminal}}$$
  Provides continuous tracking guidance, penalizes control chatter and excessive AoA, and awards substantial terminal bonuses ($+1000$) for soft landings while penalizing crashes ($-1000$).
- **Dual Engine Architecture**: Supports both official MATLAB Reinforcement Learning Toolbox (`rlTD3Agent`, `rlSACAgent`) and a standalone pure-MATLAB deep continuous actor-critic engine that runs without external toolboxes.

---

## 4. Atmospheric Disturbance Modeling

1. **Steady Winds**: Horizontal headwind/tailwind ($w_x$) and vertical currents ($w_h$).
2. **Atmospheric Wind Shear**: Logarithmic boundary-layer profile:
   $$w_x(h) = -V_{\text{ref}} \frac{\ln(h/z_0 + 1)}{\ln(h_{\text{ref}}/z_0 + 1)}$$
3. **Discrete Gusts**: MIL-F-8785C / FAA standard 1-cosine gusts with configurable amplitude and duration.
4. **Stochastic Turbulence**: Low-altitude Dryden continuous turbulence model (MIL-F-8785C) excited with reproducible pseudo-random seeds.

---

## 5. Repository Structure

```
AircraftRL/
├── main.m                         % Top-level orchestrator & interactive menu
├── aircraftParameters.m           % Single source of truth for all parameters (SI units)
├── aircraftDynamics.m             % Nonlinear longitudinal equations of motion
├── initializeAircraft.m           % Analytical & iterative trim solver
├── updateAircraft.m               % 4th-order Runge-Kutta numerical integration engine
├── landingScenario.m              % Reference generation, flare profile & touchdown logic
├── windModel.m                    % Atmospheric wind, shear, gust & Dryden turbulence
├── sensorModel.m                  % Pitot, altimeter, and IMU sensor noise & filtering
├── rewardFunction.m               % RL reward function with shaping & terminal bonuses
├── metrics.m                      % Research-grade flight & touchdown performance metrics
├── runMonteCarlo.m                % 8-suite automated Monte Carlo evaluation campaign
├── plotAircraft.m                 % Single-frame aircraft attitude visualizer
├── plotResults.m                  % Publication-grade multi-panel figure generation
├── animateFlight.m                % 2D interactive flight animation with cockpit HUD
├── Controllers/
│   ├── pidLandingController.m     % Multi-loop PID with anti-windup & flare law
│   ├── simulatePID.m              % Closed-loop PID simulation runner
│   └── tunePID.m                  % Linearization, mode analysis & gain verification
├── RL/
│   ├── AircraftLandingEnv.m       % Standalone & RL Toolbox compatible environment class
│   ├── createEnvironment.m        % Environment factory function
│   ├── rlConfig.m                 % Hyperparameters configuration
│   ├── rlEngine.m                 % Standalone pure-MATLAB continuous actor-critic engine
│   ├── createTD3Agent.m           % Twin Delayed DDPG constructor
│   ├── createSACAgent.m           % Soft Actor-Critic constructor
│   ├── trainAgent.m               % Training pipeline (smoke-test & full mode)
│   └── evaluateAgent.m            % Deterministic policy evaluation
├── Tests/
│   ├── testAircraftDynamics.m     % Unit tests for equations of motion & stall
│   ├── testPID.m                  % Unit tests for PID tracking & gusts
│   ├── testEnvironment.m          % Unit tests for RL environment & reward
│   ├── testLandingLogic.m         % Unit tests for touchdown classifications
│   └── runAllTests.m              % Master automated test suite runner
├── Results/
│   ├── Figures/                   % High-resolution publication plots & animations
│   └── Data/
│       ├── checkpoints/           % Saved RL neural network weights
│       ├── monte_carlo_results.mat% Full Monte Carlo dataset
│       ├── monte_carlo_summary.csv% Tabular summary for paper
│       └── exportPaperTables.m    % Formatter exporting LaTeX tables
├── Python/
│   ├── validate_physics.py        % Independent cross-validation suite in SciPy
│   └── test_suite.py              % Full Python test harness (all tests passing)
└── Simulink/
    ├── aircraft_model.slx         % Simulink flight dynamics model
    └── README.md                  % Complete engineering wiring & validation specification
```

---

## 6. How to Run

### 6.1 Prerequisites
- **MATLAB**: R2020a through R2026a (tested and validated).
- **Toolbox Requirements**:
  - *Base MATLAB*: Fully supports the core aircraft simulation, trim, RK4 integrator, classical PID baseline, Monte Carlo benchmarking, standalone neural RL engine, and publication plots.
  - *Reinforcement Learning Toolbox* (Optional): Automatically detected and utilized by `createTD3Agent.m` and `createSACAgent.m` if installed. If absent, the standalone pure-MATLAB engine executes transparently.
- **Python (Optional for cross-validation)**: Python 3.8+ with `numpy`, `scipy`, `matplotlib`.

### 6.2 Execution in MATLAB
Launch MATLAB, navigate to `AircraftRL`, and run `main`:

```matlab
>> main
```

An interactive menu provides 9 execution options:

1. **Basic Trim Simulation**: Verifies steady equilibrium descent on a $3^\circ$ glide slope.
2. **Classical PID Landing**: Runs approach, flare, and touchdown, printing performance metrics and displaying trajectory plots.
3. **RL Training**:
   - `trainAgent("test", "td3")`: Quick 3-episode smoke test verifying network forward/backward passes and checkpointing.
   - `trainAgent("train", "td3")`: Full training campaign (600 episodes) with domain randomization.
4. **RL Evaluation**: Deterministically simulates the latest trained checkpoint.
5. **PID vs RL Comparison**: Side-by-side trajectory, tracking error, and control action plots.
6. **Monte Carlo Campaign**: Runs independent trials across 8 disturbance and dispersion test batteries, generating CSV summaries and LaTeX tables.
7. **Generate Publication Figures**: Exports high-resolution figures to `Results/Figures/`.
8. **Flight Animation**: Interactive animation with pitch visualization and cockpit telemetry HUD.
9. **Automated Test Suite**: Executes all unit tests in `Tests/`.

Alternatively, run specific workflows directly from the MATLAB command window:

```matlab
% Run complete automated test harness
>> runAllTests

% Run closed-loop PID landing simulation
>> [flight_log, m] = simulatePID();

% Run 50 Monte Carlo trials per experimental suite
>> summary = runMonteCarlo(50);

% Animate flight at 3x speed
>> animateFlight(flight_log, aircraftParameters(), 3.0);
```

### 6.3 Independent Python Cross-Validation
To cross-validate the physical equations and control responses outside of MATLAB:

```bash
python Python/validate_physics.py
python Python/test_suite.py
```

---

## 7. Experimental Results & Verification

### 7.1 Flight Dynamics Trim Equilibrium
Verification across the nonlinear model confirms machine-precision force and moment balance at nominal approach ($V = 35.0\text{ m/s}$, $\gamma = -3.0^\circ$):
- $\dot{V} = 0.00000\text{ m/s}^2$
- $\dot{\gamma} = 0.00000\text{ rad/s}^2$
- $\dot{q} = 0.00000\text{ rad/s}^2$
- Elevator trim: $\delta_{e,\text{trim}} = -4.96^\circ$, Throttle trim: $\delta_{T,\text{trim}} = 10.7\%$

### 7.2 Closed-Loop PID Baseline Performance
Under nominal conditions and boundary-layer wind shear:
- **Touchdown Position**: $x_{\text{td}} = 291.6\text{ m}$ (Aim point: $300.0\text{ m}$, error: $-8.4\text{ m}$).
- **Touchdown Sink Rate**: $\dot{h}_{\text{td}} = 1.53\text{ m/s}$ (Safe soft limit $\le 1.8\text{ m/s}$).
- **Touchdown Pitch**: $\theta_{\text{td}} = 6.3^\circ$ (Nose-up main gear contact, tail clearance maintained).
- **RMS Glide-Slope Error**: $1.84\text{ m}$.
- **Status**: `SOFT_TOUCHDOWN` (Passed all ICAO safety criteria).

---

## 8. Technical Limitations & Honest Reporting

1. **Lateral-Directional Dynamics**: The present model focuses rigorously on 6-state longitudinal motion ($x, h, V, \gamma, \theta, q$). Crosswind lateral drift and roll/yaw dynamics are not included and represent a natural next research extension.
2. **Ground Contact Physics**: Ground reaction forces are represented by altitude clamping and flight-path ground alignment upon touchdown; oleo-pneumatic landing gear spring-damper dynamics are not modeled in continuous time.
3. **Simulink Implementation**: The Simulink model file (`Simulink/aircraft_model.slx`) contains the baseline layout and is to be completed manually per the detailed specification in [`Simulink/README.md`](Simulink/README.md).


## Copyright

Copyright © 2026 Shreyan Dey. All rights reserved.

This repository is provided for academic, research, and portfolio
reference purposes. No permission is granted to reproduce, modify,
distribute, or commercially use this code without prior written
permission from the author.

© 2026 Shreyan Dey
