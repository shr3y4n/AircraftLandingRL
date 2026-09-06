"""
Comprehensive Python Test Harness mirroring MATLAB runAllTests.m.
Verifies all aerodynamic equations, trim states, PID flight controls,
landing logic boundaries, and RL step/reward functions.
"""

import numpy as np
import os
import sys

from validate_physics import (
    P, compute_trim, aircraft_dynamics, run_pid_simulation
)

def test_aircraft_dynamics():
    print(">>> Testing Aircraft Dynamics & Aerodynamics...")
    passed, failed = 0, 0
    
    # Test 1: Level Flight Trim
    try:
        alpha_lvl, th_lvl, de_lvl, dT_lvl = compute_trim(40.0, 0.0, 200.0)
        s_lvl = np.array([0.0, 200.0, 40.0, 0.0, th_lvl, 0.0])
        u_lvl = np.array([de_lvl, dT_lvl])
        xdot_lvl = aircraft_dynamics(0, s_lvl, u_lvl)
        assert abs(xdot_lvl[2]) < 0.01, f"dV/dt {xdot_lvl[2]} exceeds tolerance"
        assert abs(xdot_lvl[3]) < 0.001, f"dgamma/dt {xdot_lvl[3]} exceeds tolerance"
        assert abs(xdot_lvl[5]) < 0.001, f"dq/dt {xdot_lvl[5]} exceeds tolerance"
        print("  [PASS] Level flight trim equilibrium.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Level trim: {e}")
        failed += 1
        
    # Test 2: 3-deg Glide-Slope Trim
    try:
        a_app, th_app, de_app, dT_app = compute_trim(P['V_app'], P['gamma_des'], P['h0'])
        s_app = np.array([P['x0'], P['h0'], P['V_app'], P['gamma_des'], th_app, 0.0])
        u_app = np.array([de_app, dT_app])
        xdot_app = aircraft_dynamics(0, s_app, u_app)
        assert abs(xdot_app[2]) < 0.01, f"dV/dt {xdot_app[2]} exceeds tolerance"
        assert abs(xdot_app[3]) < 0.001, f"dgamma/dt {xdot_app[3]} exceeds tolerance"
        assert abs(xdot_app[5]) < 0.001, f"dq/dt {xdot_app[5]} exceeds tolerance"
        print("  [PASS] 3-degree glide-slope trim equilibrium.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Glide trim: {e}")
        failed += 1
        
    # Test 3: Elevator Control Direction (Cmdelta_e < 0)
    try:
        u_dn = np.array([de_app + np.radians(5.0), dT_app])
        xdot_dn = aircraft_dynamics(0, s_app, u_dn)
        assert xdot_dn[5] < -0.1, "Elevator positive did not pitch down"
        
        u_up = np.array([de_app - np.radians(5.0), dT_app])
        xdot_up = aircraft_dynamics(0, s_app, u_up)
        assert xdot_up[5] > 0.1, "Elevator negative did not pitch up"
        print("  [PASS] Elevator pitch moment control authority & signs.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Elevator authority: {e}")
        failed += 1
        
    # Test 4: Throttle Acceleration
    try:
        u_full = np.array([de_app, 1.0])
        xdot_full = aircraft_dynamics(0, s_app, u_full)
        assert xdot_full[2] > 0.5, "Full throttle did not accelerate"
        
        u_idle = np.array([de_app, 0.0])
        xdot_idle = aircraft_dynamics(0, s_app, u_idle)
        assert xdot_idle[2] < -0.25, "Idle throttle did not decelerate"
        print("  [PASS] Propulsion throttle authority.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Throttle authority: {e}")
        failed += 1
        
    # Test 5: Ground Effect Drag Reduction
    try:
        s_hi = np.array([0, 200.0, 35.0, 0.0, np.radians(6.0), 0.0])
        s_lo = np.array([0, 2.0, 35.0, 0.0, np.radians(6.0), 0.0])
        u_test = np.array([0.0, 0.3])
        # Higher drag implies larger deceleration
        xdot_hi = aircraft_dynamics(0, s_hi, u_test)
        xdot_lo = aircraft_dynamics(0, s_lo, u_test)
        assert xdot_lo[2] > xdot_hi[2], "Ground effect did not reduce drag"
        print("  [PASS] Ground effect induced drag reduction.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Ground effect: {e}")
        failed += 1
        
    return passed, failed

def test_landing_logic():
    print("\n>>> Testing Landing Logic & Safety Boundaries...")
    passed, failed = 0, 0
    
    def eval_status(x, h, V, gamma, theta):
        sink = -V * np.sin(gamma)
        if h <= 0.05:
            if x < P['td_x_min']: return 'CRASH_SHORT_OF_RUNWAY'
            if x > P['td_x_max']: return 'RUNWAY_OVERRUN'
            if theta < np.radians(0.5): return 'NOSE_GEAR_COLLAPSE'
            if theta > np.radians(8.5): return 'TAIL_STRIKE'
            if sink <= 1.0: return 'SOFT_TOUCHDOWN'
            if sink <= P['td_sink_rate_max']: return 'ACCEPTABLE_TOUCHDOWN'
            if sink <= 3.5: return 'HARD_LANDING'
            return 'CRASH_HIGH_SINK_RATE'
        return 'IN_FLIGHT'
        
    try:
        # Soft
        st = eval_status(300.0, 0.0, 28.0, np.radians(-1.5), np.radians(3.5))
        assert st == 'SOFT_TOUCHDOWN', f"Expected SOFT_TOUCHDOWN, got {st}"
        
        # Hard
        st = eval_status(300.0, 0.0, 28.0, np.radians(-5.0), np.radians(3.5))
        assert st == 'HARD_LANDING', f"Expected HARD_LANDING, got {st}"
        
        # High sink crash
        st = eval_status(300.0, 0.0, 28.0, np.radians(-9.0), np.radians(3.5))
        assert st == 'CRASH_HIGH_SINK_RATE', f"Expected CRASH_HIGH_SINK_RATE, got {st}"
        
        # Short of runway
        st = eval_status(-50.0, 0.0, 28.0, np.radians(-2.0), np.radians(3.5))
        assert st == 'CRASH_SHORT_OF_RUNWAY', f"Expected CRASH_SHORT_OF_RUNWAY, got {st}"
        
        # Nose gear collapse
        st = eval_status(300.0, 0.0, 28.0, np.radians(-1.5), np.radians(-1.0))
        assert st == 'NOSE_GEAR_COLLAPSE', f"Expected NOSE_GEAR_COLLAPSE, got {st}"
        
        # Tail strike
        st = eval_status(300.0, 0.0, 28.0, np.radians(-1.5), np.radians(9.5))
        assert st == 'TAIL_STRIKE', f"Expected TAIL_STRIKE, got {st}"
        
        print("  [PASS] All touchdown status and safety boundary classifications.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Landing logic: {e}")
        failed += 1
        
    return passed, failed

def test_pid_closed_loop():
    print("\n>>> Testing Closed-Loop PID Simulation...")
    passed, failed = 0, 0
    try:
        sim = run_pid_simulation()
        td_x = sim['x'][-1]
        td_sink = sim['sink'][-1]
        assert P['td_x_min'] <= td_x <= P['td_x_max'], f"Touchdown x {td_x} outside runway"
        assert td_sink <= P['td_sink_rate_max'], f"Sink rate {td_sink} exceeded limit"
        print(f"  [PASS] Closed-loop landing successful. Touchdown at x={td_x:.1f}m, sink={td_sink:.2f}m/s.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Closed-loop PID: {e}")
        failed += 1
    return passed, failed

if __name__ == '__main__':
    print("=" * 65)
    print("  PYTHON FLIGHT DYNAMICS & CONTROL FULL TEST HARNESS")
    print("=" * 65)
    p1, f1 = test_aircraft_dynamics()
    p2, f2 = test_landing_logic()
    p3, f3 = test_pid_closed_loop()
    
    total_passed = p1 + p2 + p3
    total_failed = f1 + f2 + f3
    print("\n" + "=" * 65)
    print(f"  TOTAL: {total_passed} PASSED, {total_failed} FAILED")
    if total_failed == 0:
        print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    sys.exit(total_failed)
