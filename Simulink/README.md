# Simulink Longitudinal Flight Dynamics & Control Model Specification

## 1. Overview
This document specifies the exact mathematical formulation, signal definitions, state conventions, and block-level architecture required to build, wire, and validate the Simulink model (`aircraft_model.slx`) against the verified MATLAB simulation framework.

The model reproduces the continuous 6-DOF longitudinal nonlinear flight dynamics of the fixed-wing utility aircraft during approach, glide-slope descent, flare, and touchdown.

---

## 2. State-Space Vector Definition & Ordering

The aircraft longitudinal state vector $\mathbf{x}(t) \in \mathbb{R}^6$ is defined as follows:

| Index | State Symbol | Definition / Description | SI Unit | Initial Condition (Nominal) |
|---|---|---|---|---|
| `x(1)` | $x$ | Longitudinal horizontal ground position along runway centerline | $\text{m}$ | $-3000.0\text{ m}$ |
| `x(2)` | $h$ | Geometric altitude above runway elevation ($h = -z_{\text{NED}}$) | $\text{m}$ | $+172.9\text{ m}$ |
| `x(3)` | $V$ | True airspeed relative to the surrounding air mass | $\text{m/s}$ | $35.0\text{ m/s}$ ($68.0\text{ kt}$) |
| `x(4)` | $\gamma$ | Air-relative flight-path angle ($\arcsin(\dot{h}_a / V_a)$) | $\text{rad}$ | $-0.05236\text{ rad}$ ($-3.0^\circ$) |
| `x(5)` | $\theta$ | Fuselage pitch attitude angle relative to horizontal | $\text{rad}$ | $+0.0858\text{ rad}$ ($+4.92^\circ$) |
| `x(6)` | $q$ | Body-axis pitch angular rate ($\dot{\theta}$) | $\text{rad/s}$ | $0.0\text{ rad/s}$ |

---

## 3. Input Signals & Physical Boundaries

### 3.1 Control Input Vector $\mathbf{u}(t) \in \mathbb{R}^2$

| Port | Signal | Definition | Range / Bounds | SI Unit | Nominal Trim Value |
|---|---|---|---|---|---|
| 1 | $\delta_e$ | Elevator surface deflection (trailing edge down is positive) | $[-0.4363, +0.2618]\text{ rad}$ ($-25^\circ \to +15^\circ$) | $\text{rad}$ | $-0.0865\text{ rad}$ ($-4.96^\circ$) |
| 2 | $\delta_T$ | Engine throttle setting fraction | $[0.0, 1.0]$ | $-$ | $0.1073$ ($10.73\%$) |

### 3.2 Atmospheric Disturbance Vector $\mathbf{w}(t) \in \mathbb{R}^4$

| Port | Signal | Definition | SI Unit | Nominal |
|---|---|---|---|---|
| 3 | $w_x$ | Horizontal longitudinal wind velocity (+tailwind, -headwind) | $\text{m/s}$ | $0.0$ |
| 4 | $w_h$ | Vertical atmospheric wind velocity (+updraft, -downdraft) | $\text{m/s}$ | $0.0$ |
| 5 | $\dot{w}_x$ | Horizontal wind acceleration component | $\text{m/s}^2$ | $0.0$ |
| 6 | $\dot{w}_h$ | Vertical wind acceleration component | $\text{m/s}^2$ | $0.0$ |

---

## 4. Governing Nonlinear Equations of Motion

### 4.1 Kinematics & Aerodynamic State Quantities
$$\alpha = \theta - \gamma \quad (\text{Angle of attack [rad]})$$
$$\bar{q} = \frac{1}{2} \rho_0 V^2 \quad (\text{Dynamic pressure } [\text{N/m}^2])$$
$$\hat{q} = \frac{\bar{c}}{2V} q \quad (\text{Non-dimensional pitch rate})$$

### 4.2 Ground Effect Model
When altitude $h < b$ (wingspan $b = 10\text{ m}$):
$$\sigma_{\text{ge}}(h) = \frac{(16 (h/b))^2}{1 + (16 (h/b))^2} \in [0.20, 1.0]$$
$$K_{\text{eff}} = K \cdot \sigma_{\text{ge}}(h), \quad K = \frac{1}{\pi \cdot e \cdot AR} \approx 0.0662$$

### 4.3 Aerodynamic Coefficients with $C^1$ Smoothstep Stall
Pre-stall linear coefficients:
$$C_{L,\text{lin}} = C_{L0} + C_{L\alpha} \alpha + C_{L\delta_e} \delta_e + C_{Lq} \hat{q}$$
$$C_m = C_{m0} + C_{m\alpha} \alpha + C_{m\delta_e} \delta_e + C_{mq} \hat{q}$$

Smoothstep stall blending factor $f_{\text{stall}} \in [0, 1]$:
$$f_{\text{pos}} = \begin{cases} 
0, & \alpha \le \alpha_{\text{stall}} - \Delta\alpha \\
3 s^2 - 2 s^3, & \alpha_{\text{stall}} - \Delta\alpha < \alpha < \alpha_{\text{stall}} + \Delta\alpha \quad \left(s = \frac{\alpha - (\alpha_{\text{stall}} - \Delta\alpha)}{2\Delta\alpha}\right) \\
1, & \alpha \ge \alpha_{\text{stall}} + \Delta\alpha
\end{cases}$$
Blended coefficients:
$$C_L = (1 - f_{\text{stall}}) C_{L,\text{lin}} + f_{\text{stall}} \left( 2 \text{sign}(\alpha) \sin^2\alpha \cos\alpha \right)$$
$$C_D = C_{D0} + K_{\text{eff}} C_L^2 + C_{D\delta_e} \delta_e^2 + f_{\text{stall}} \left( 2 \sin^3|\alpha| \right)$$

### 4.4 Dimensional Forces and Pitching Moment
$$L = \bar{q} S C_L \quad (\text{Aerodynamic Lift [N]})$$
$$D = \bar{q} S C_D \quad (\text{Aerodynamic Drag [N]})$$
$$T = \delta_T \cdot T_{\text{max}} \quad (\text{Engine Thrust [N]})$$
$$M = \bar{q} S \bar{c} C_m \quad (\text{Pitching Moment [N}\cdot\text{m]})$$

### 4.5 Continuous Differential Equations ($\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}, \mathbf{w})$)
$$\dot{x} = V \cos\gamma + w_x$$
$$\dot{h} = V \sin\gamma + w_h$$
$$\dot{V} = \frac{T \cos\alpha - D}{m} - g \sin\gamma - (\dot{w}_x \cos\gamma + \dot{w}_h \sin\gamma)$$
$$\dot{\gamma} = \frac{T \sin\alpha + L}{m V} - \frac{g \cos\gamma}{V} + \frac{\dot{w}_x \sin\gamma - \dot{w}_h \cos\gamma}{V}$$
$$\dot{\theta} = q$$
$$\dot{q} = \frac{M}{I_{yy}}$$

---

## 5. Simulink Block-Level Architecture

When finishing or building `aircraft_model.slx`, organize into 4 modular subsystems:

```
+-------------------------------------------------------------------------------+
|                             AIRCRAFT_MODEL.SLX                                |
|                                                                               |
|  Controls [de, dT] ----> [ Subsystem 1: Aerodynamics & Thrust ]               |
|  Wind [wx, wh] -------->  - Dynamic pressure qbar                             |
|  Feedback State x ----->  - Stall blending f_stall                            |
|                           - Lift L, Drag D, Moment M, Thrust T                |
|                                       |                                       |
|                                       v                                       |
|                          [ Subsystem 2: Equations of Motion ]                 |
|                           - Evaluates [xdot, hdot, Vdot, gammadot, thetadot, qdot]
|                                       |                                       |
|                                       v                                       |
|                          [ Subsystem 3: 6x Integrators ]                      |
|                           - Integrator 1: x   (IC = -3000)                    |
|                           - Integrator 2: h   (IC = 172.9, min = 0)           |
|                           - Integrator 3: V   (IC = 35.0)                     |
|                           - Integrator 4: gam (IC = -0.05236)                 |
|                           - Integrator 5: the (IC = 0.0858)                   |
|                           - Integrator 6: q   (IC = 0.0)                      |
|                                       |                                       |
|                                       v                                       |
|                          [ Subsystem 4: Sensor & Telemetry Out ]              |
|                           - Ground speed, sink rate (-hdot), alpha            |
|                           - Outports / Scopes                                 |
+-------------------------------------------------------------------------------+
```

### 5.1 Integrator Configuration
- Solver: Fixed-step `ode4` (Runge-Kutta 4) or variable-step `ode45` (Dormand-Prince).
- Fixed-step size: `0.02` seconds ($50\text{ Hz}$).
- Integrator 2 ($h$): Check *Limit output*, Lower saturation limit = `0.0`.

---

## 6. Verification & Step-by-Step Validation Against MATLAB

To guarantee identical numerical behavior between Simulink and the MATLAB `updateAircraft.m` engine:

1. **Load Parameters**: Run `P = aircraftParameters(); [s0, trim] = initializeAircraft(P);` in the MATLAB base workspace.
2. **Apply Constant Trim Inputs**:
   - Set Elevator constant block = `trim.delta_e` ($-0.0865\text{ rad}$).
   - Set Throttle constant block = `trim.delta_T` ($0.1073$).
   - Set Wind = `[0; 0]`.
3. **Simulate for $10\text{ seconds}$**:
   - Verify that $\dot{V} \approx 0.0000$, $\dot{\gamma} \approx 0.0000$, $\dot{q} \approx 0.0000$.
   - Altitude should descend steadily at exactly $V \sin(-3^\circ) \approx -1.8318\text{ m/s}$.
4. **Step Response Check**:
   - Inject a $+1^\circ$ ($+0.01745\text{ rad}$) elevator step at $t = 2.0\text{ s}$.
   - Confirm that pitch rate $q$ goes negative (nose pitches down) and dampens back to a trimmed descent.
   - Max numerical state discrepancy between Simulink and MATLAB RK4 must be $< 10^{-5}$.
