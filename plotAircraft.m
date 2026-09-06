function plotAircraft(state, P)
% PLOTAIRCRAFT Quick visual inspection of aircraft position, attitude, and runway layout.
%
% Renders the runway surface, threshold, glide-slope reference vector,
% and the aircraft orientation corresponding to pitch attitude theta.
%
% Inputs:
%   state - Aircraft state struct or vector [x; h; V; gamma; theta; q]
%   P     - Parameters struct (optional)

if nargin < 2 || isempty(P)
    P = aircraftParameters();
end

if isstruct(state)
    x     = state.x;
    h     = state.h;
    theta = state.theta;
    V     = state.V;
    gamma = state.gamma;
else
    x     = state(1);
    h     = state(2);
    V     = state(3);
    gamma = state(4);
    theta = state(5);
end

clf;
hold on; grid on; box on;

% Runway Ground Line & Aim Point
plot([-500, 1500], [0, 0], 'k-', 'LineWidth', 2.5);
plot([0, 0], [-2, 15], 'Color', [0 0.6 0], 'LineWidth', 2);
plot(P.touchdown_aim_x, 0, 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 12);

% Glide-Slope Line
x_gs = linspace(-3200, P.touchdown_aim_x, 100);
h_gs = -(x_gs - P.touchdown_aim_x) * tan(-P.gamma_des);
plot(x_gs, h_gs, 'k--', 'LineWidth', 1.2);

% Aircraft Representation with Authentic Pitch Attitude theta
fuselage_len = 25.0; % Scaled for visibility on runway scale
dx_fuse = fuselage_len * cos(theta);
dh_fuse = fuselage_len * sin(theta);

% Draw fuselage line
plot([x - dx_fuse/2, x + dx_fuse/2], [h - dh_fuse/2, h + dh_fuse/2], ...
     'b-', 'LineWidth', 3.5);

% Draw wing / CG marker
plot(x, h, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% Draw velocity vector arrow
v_scale = 1.0;
quiver(x, h, V * cos(gamma) * v_scale, V * sin(gamma) * v_scale, 0, ...
       'Color', [0.8 0.2 0.2], 'LineWidth', 1.5, 'MaxHeadSize', 0.8);

xlabel('Distance x [m]', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Altitude h [m]', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('Aircraft Position: x=%.1fm, h=%.1fm, V=%.1fm/s, \\theta=%.1f^\\circ, \\gamma=%.1f^\\circ', ...
      x, h, V, rad2deg(theta), rad2deg(gamma)), 'FontSize', 11);

xlim([x - 300, x + 300]);
ylim([max(-5, h - 50), h + 100]);

drawnow;

end