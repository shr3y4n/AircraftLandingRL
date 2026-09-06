"""
Python Physics & Control Cross-Validation Suite for Aircraft Landing RL Framework.
Validates nonlinear flight equations of motion, trim states, and PID tracking.
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import os

# 1. Aircraft Parameters (Identical to aircraftParameters.m)
P = {
    'mass': 1100.0,
    'Iyy': 1750.0,
    'S': 16.2,
    'b': 10.0,
    'c': 1.62,
    'AR': 10.0**2 / 16.2,
    'rho0': 1.225,
    'g': 9.80665,
    'CL0': 0.28,
    'CLalpha': 4.58,
    'CLq': 3.80,
    'CLdelta_e': 0.36,
    'CD0': 0.028,
    'e': 0.78,
    'CDdelta_e': 0.04,
    'Cm0': 0.00,
    'Cmalpha': -0.72,
    'Cmq': -9.00,
    'Cmdelta_e': -1.15,
    'alpha_stall_pos': np.radians(14.0),
    'alpha_stall_neg': np.radians(-10.0),
    'CL_max': 1.45,
    'stall_transition': np.radians(2.0),
    'ground_effect_enabled': True,
    'maxThrust': 3800.0,
    'elevator_min': np.radians(-25.0),
    'elevator_max': np.radians(15.0),
    'throttle_min': 0.0,
    'throttle_max': 1.0,
    'V_app': 35.0,
    'V_td': 28.0,
    'gamma_des': np.radians(-3.0),
    'h0': 172.9,
    'x0': -3000.0,
    'touchdown_aim_x': 300.0,
    'h_flare': 12.0,
    'hdot_flare_td': -0.50,
    'theta_flare_tgt': np.radians(3.5),
    'td_sink_rate_max': 1.8,
    'td_x_min': 0.0,
    'td_x_max': 650.0,
    'dt': 0.02
}
P['K'] = 1.0 / (np.pi * P['e'] * P['AR'])

# 2. Trim Calculation
def compute_trim(V0, gamma0, h0):
    qbar = 0.5 * P['rho0'] * V0**2
    mg = P['mass'] * P['g']
    
    K_eff = P['K']
    if P['ground_effect_enabled']:
        h_eff = max(0.5, h0)
        h_b = h_eff / P['b']
        sigma_ge = (16.0 * h_b**2) / (1.0 + 16.0 * h_b**2)
        K_eff = P['K'] * max(0.2, min(1.0, sigma_ge))
        
    L_target = mg * np.cos(gamma0)
    CL_target = L_target / (qbar * P['S'])
    
    dCL_dalpha = P['CLalpha'] - (P['CLdelta_e'] * P['Cmalpha'] / P['Cmdelta_e'])
    CL0_trim = P['CL0'] - (P['CLdelta_e'] * P['Cm0'] / P['Cmdelta_e'])
    
    alpha_trim = (CL_target - CL0_trim) / dCL_dalpha
    delta_e_trim = -(P['Cm0'] + P['Cmalpha'] * alpha_trim) / P['Cmdelta_e']
    
    for _ in range(5):
        CL = P['CL0'] + P['CLalpha'] * alpha_trim + P['CLdelta_e'] * delta_e_trim
        CD = P['CD0'] + K_eff * CL**2 + P['CDdelta_e'] * delta_e_trim**2
        Drag = qbar * P['S'] * CD
        T_req = (Drag + mg * np.sin(gamma0)) / np.cos(alpha_trim)
        T_req = np.clip(T_req, 0.0, P['maxThrust'])
        L_req = mg * np.cos(gamma0) - T_req * np.sin(alpha_trim)
        CL_req = L_req / (qbar * P['S'])
        alpha_trim += (CL_req - CL) / dCL_dalpha
        delta_e_trim = -(P['Cm0'] + P['Cmalpha'] * alpha_trim) / P['Cmdelta_e']
        
    delta_T_trim = np.clip(T_req / P['maxThrust'], 0.0, 1.0)
    delta_e_trim = np.clip(delta_e_trim, P['elevator_min'], P['elevator_max'])
    theta_trim = gamma0 + alpha_trim
    return alpha_trim, theta_trim, delta_e_trim, delta_T_trim

# 3. Nonlinear Dynamics
def aircraft_dynamics(t, state, controls, wind=[0.0, 0.0, 0.0, 0.0]):
    x, h, V, gamma, theta, q = state
    delta_e, delta_T = controls
    wx, wh, dwx_dt, dwh_dt = wind
    
    V = max(5.0, V)
    delta_e = np.clip(delta_e, P['elevator_min'], P['elevator_max'])
    delta_T = np.clip(delta_T, 0.0, 1.0)
    
    alpha = theta - gamma
    qbar = 0.5 * P['rho0'] * V**2
    
    K_eff = P['K']
    if P['ground_effect_enabled']:
        h_eff = max(0.5, h)
        h_b = h_eff / P['b']
        sigma_ge = (16.0 * h_b**2) / (1.0 + 16.0 * h_b**2)
        K_eff = P['K'] * max(0.2, min(1.0, sigma_ge))
        
    q_hat = (P['c'] / (2.0 * V)) * q
    CL_lin = P['CL0'] + P['CLalpha'] * alpha + P['CLdelta_e'] * delta_e + P['CLq'] * q_hat
    
    da = P['stall_transition']
    f_pos = 0.0
    if alpha > (P['alpha_stall_pos'] + da):
        f_pos = 1.0
    elif alpha > (P['alpha_stall_pos'] - da):
        s = (alpha - (P['alpha_stall_pos'] - da)) / (2.0 * da)
        f_pos = 3.0 * s**2 - 2.0 * s**3
        
    f_neg = 0.0
    if alpha < (P['alpha_stall_neg'] - da):
        f_neg = 1.0
    elif alpha < (P['alpha_stall_neg'] + da):
        s = ((P['alpha_stall_neg'] + da) - alpha) / (2.0 * da)
        f_neg = 3.0 * s**2 - 2.0 * s**3
        
    f_stall = min(1.0, f_pos + f_neg)
    
    CL_sep = 2.0 * np.sign(alpha) * (np.sin(alpha)**2) * np.cos(alpha)
    CL = (1.0 - f_stall) * CL_lin + f_stall * CL_sep
    CD = P['CD0'] + K_eff * CL**2 + P['CDdelta_e'] * delta_e**2 + f_stall * (2.0 * np.sin(abs(alpha))**3)
    Cm = P['Cm0'] + P['Cmalpha'] * alpha + P['Cmdelta_e'] * delta_e + P['Cmq'] * q_hat
    if f_stall > 0.01:
        Cm -= 0.5 * f_stall * np.sign(alpha)
        
    Lift = qbar * P['S'] * CL
    Drag = qbar * P['S'] * CD
    Thrust = delta_T * P['maxThrust']
    M_pitch = qbar * P['S'] * P['c'] * Cm
    
    xdot = V * np.cos(gamma) + wx
    hdot = V * np.sin(gamma) + wh
    Vdot = (Thrust * np.cos(alpha) - Drag) / P['mass'] - P['g'] * np.sin(gamma) - (dwx_dt * np.cos(gamma) + dwh_dt * np.sin(gamma))
    gammadot = (Thrust * np.sin(alpha) + Lift) / (P['mass'] * V) - (P['g'] * np.cos(gamma)) / V + (dwx_dt * np.sin(gamma) - dwh_dt * np.cos(gamma)) / V
    thetadot = q
    qdot = M_pitch / P['Iyy']
    
    return np.array([xdot, hdot, Vdot, gammadot, thetadot, qdot])

# 4. PID Landing Simulation in Python
def run_pid_simulation():
    alpha_trim, theta_trim, de_trim, dT_trim = compute_trim(P['V_app'], P['gamma_des'], P['h0'])
    state = np.array([P['x0'], P['h0'], P['V_app'], P['gamma_des'], theta_trim, 0.0])
    
    dt = P['dt']
    T_max = 100.0
    N = int(T_max / dt)
    
    int_theta = 0.0
    int_V = 0.0
    q_filt = 0.0
    prev_de = de_trim
    prev_dT = dT_trim
    
    log = {'t': [], 'x': [], 'h': [], 'V': [], 'gamma': [], 'theta': [], 'q': [], 'sink': [], 'de': [], 'dT': [], 'href': []}
    
    for step in range(N):
        t = step * dt
        x, h, V, gamma, theta, q = state
        sink_rate = -V * np.sin(gamma)
        
        # Touchdown check
        if h <= 0.05:
            log['status'] = 'TOUCHDOWN'
            break
            
        # Reference
        h_gs = max(0.0, -(x - P['touchdown_aim_x']) * np.tan(-P['gamma_des']))
        if h > P['h_flare']:
            h_ref = h_gs
            hdot_ref = P['V_app'] * np.sin(P['gamma_des'])
            V_ref = P['V_app']
            e_h = h_ref - h
            e_hdot = hdot_ref - (-sink_rate)
            theta_cmd = theta_trim + 0.022 * e_h + 0.035 * e_hdot
            theta_cmd = np.clip(theta_cmd, np.radians(-8.0), np.radians(10.0))
        else:
            flare_ratio = max(0.0, min(1.0, h / P['h_flare']))
            hdot_ref = P['hdot_flare_td'] + (P['V_app'] * np.sin(P['gamma_des']) - P['hdot_flare_td']) * flare_ratio
            h_ref = P['h_flare'] * (flare_ratio**1.5)
            V_ref = P['V_td'] + (P['V_app'] - P['V_td']) * flare_ratio
            e_hdot = hdot_ref - (-sink_rate)
            theta_cmd = np.radians(3.2) + 0.055 * e_hdot
            theta_cmd = np.clip(theta_cmd, np.radians(-2.0), np.radians(10.0))
            
        # Pitch inner loop
        e_theta = theta_cmd - theta
        q_filt += dt * 30.0 * (q - q_filt)
        de_unsat = de_trim - (2.40 * e_theta + 0.35 * int_theta - 0.65 * q)
        delta_e = np.clip(de_unsat, P['elevator_min'], P['elevator_max'])
        if delta_e == de_unsat:
            int_theta = np.clip(int_theta + e_theta * dt, np.radians(-15), np.radians(15))
            
        # Airspeed loop
        e_V = V_ref - V
        dT_unsat = dT_trim + 0.085 * e_V + 0.018 * int_V
        delta_T = np.clip(dT_unsat, 0.0, 1.0)
        if delta_T == dT_unsat:
            int_V = np.clip(int_V + e_V * dt, -5.0, 5.0)
            
        # RK4
        controls = np.array([delta_e, delta_T])
        k1 = aircraft_dynamics(t, state, controls)
        k2 = aircraft_dynamics(t + 0.5*dt, state + 0.5*dt*k1, controls)
        k3 = aircraft_dynamics(t + 0.5*dt, state + 0.5*dt*k2, controls)
        k4 = aircraft_dynamics(t + dt, state + dt*k3, controls)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if state[1] < 0: state[1] = 0.0
        
        log['t'].append(t)
        log['x'].append(x)
        log['h'].append(h)
        log['V'].append(V)
        log['gamma'].append(gamma)
        log['theta'].append(theta)
        log['q'].append(q)
        log['sink'].append(sink_rate)
        log['de'].append(delta_e)
        log['dT'].append(delta_T)
        log['href'].append(h_ref)
        
    for k in log:
        if isinstance(log[k], list):
            log[k] = np.array(log[k])
    return log

if __name__ == '__main__':
    print("=" * 60)
    print("  PYTHON CROSS-VALIDATION OF FLIGHT DYNAMICS & CONTROL")
    print("=" * 60)
    
    alpha_trim, theta_trim, de_trim, dT_trim = compute_trim(P['V_app'], P['gamma_des'], P['h0'])
    print(f"Trim calculated:")
    print(f"  alpha_trim   = {np.degrees(alpha_trim):.2f} deg")
    print(f"  theta_trim   = {np.degrees(theta_trim):.2f} deg")
    print(f"  delta_e_trim = {np.degrees(de_trim):.2f} deg")
    print(f"  delta_T_trim = {dT_trim*100:.1f} %")
    
    # Check trim equilibrium derivative
    state_0 = np.array([P['x0'], P['h0'], P['V_app'], P['gamma_des'], theta_trim, 0.0])
    ctrl_0 = np.array([de_trim, dT_trim])
    xdot0 = aircraft_dynamics(0, state_0, ctrl_0)
    print(f"Equilibrium state derivatives at trim:")
    print(f"  dV/dt      = {xdot0[2]:.5f} m/s^2 (target ~ 0)")
    print(f"  dgamma/dt  = {xdot0[3]:.5f} rad/s^2 (target ~ 0)")
    print(f"  dq/dt      = {xdot0[5]:.5f} rad/s^2 (target ~ 0)")
    assert abs(xdot0[2]) < 0.02, "dV/dt trim error too high"
    assert abs(xdot0[3]) < 0.005, "dgamma/dt trim error too high"
    assert abs(xdot0[5]) < 0.005, "dq/dt trim error too high"
    print("  --> Trim verified successfully!")
    
    # Run full trajectory
    print("\nRunning closed-loop approach, flare, and touchdown simulation...")
    sim_log = run_pid_simulation()
    
    td_x = sim_log['x'][-1]
    td_sink = sim_log['sink'][-1]
    td_V = sim_log['V'][-1]
    td_pitch = np.degrees(sim_log['theta'][-1])
    
    print(f"Touchdown Results:")
    print(f"  Longitudinal x:    {td_x:.1f} m (Aim: {P['touchdown_aim_x']:.1f} m)")
    print(f"  Vertical Sink:     {td_sink:.2f} m/s (Limit: <= {P['td_sink_rate_max']:.1f} m/s)")
    print(f"  Touchdown Speed:   {td_V:.1f} m/s (Target: {P['V_td']:.1f} m/s)")
    print(f"  Touchdown Pitch:   {td_pitch:.1f} deg (Target: {np.degrees(P['theta_flare_tgt']):.1f} deg)")
    
    assert 0.0 <= td_x <= 650.0, "Touchdown outside runway zone"
    assert td_sink <= P['td_sink_rate_max'], "Touchdown sink rate exceeded structural limit"
    assert 0.5 <= td_pitch <= 8.5, "Touchdown pitch attitude unsafe"
    print("  --> Soft Touchdown & Landing Criteria Passed Successfully!")
    
    # Save verification figure
    os.makedirs('Results/Figures', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(sim_log['x'], sim_log['href'], 'k--', label='Reference Profile')
    axes[0, 0].plot(sim_log['x'], sim_log['h'], 'b-', label='Aircraft Trajectory')
    axes[0, 0].axhline(0, color='gray', lw=2)
    axes[0, 0].set_xlabel('Distance x [m]')
    axes[0, 0].set_ylabel('Altitude h [m]')
    axes[0, 0].set_title('Approach & Flare Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(sim_log['t'], sim_log['V'], 'g-', label='Airspeed V')
    axes[0, 1].axhline(P['V_app'], color='k', ls='--', label='V_app')
    axes[0, 1].axhline(P['V_td'], color='r', ls=':', label='V_td')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Airspeed [m/s]')
    axes[0, 1].set_title('Airspeed Response')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(sim_log['t'], np.degrees(sim_log['theta']), 'b-', label='Pitch Attitude')
    axes[1, 0].plot(sim_log['t'], np.degrees(sim_log['gamma']), 'r--', label='Flight-Path Angle')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Angle [deg]')
    axes[1, 0].set_title('Attitude & Flight Path')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(sim_log['t'], sim_log['sink'], 'm-', label='Sink Rate')
    axes[1, 1].axhline(P['td_sink_rate_max'], color='r', ls=':', label='Max Safe Limit')
    axes[1, 1].axhline(1.0, color='g', ls='--', label='Soft Landing')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Sink Rate [m/s]')
    axes[1, 1].set_title('Descent & Flare Sink Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('Results/Figures/python_physics_validation.png', dpi=150)
    print("Saved validation figure to Results/Figures/python_physics_validation.png")
    print("=" * 60)
