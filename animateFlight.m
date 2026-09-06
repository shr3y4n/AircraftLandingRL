function animateFlight(flight_log, P, playback_speed, save_video)
% ANIMATEFLIGHT Interactive 2D flight animation with authentic aircraft attitude and HUD.
%
% Visualizes approach, flare, and touchdown with tracking camera, glide-slope corridor,
% pitch attitude orientation, and real-time cockpit telemetry HUD.
%
% Inputs:
%   flight_log     - Flight telemetry struct from simulatePID or evaluateAgent
%   P              - Aircraft parameters struct (optional)
%   playback_speed - Playback multiplier (e.g. 2.0x, 5.0x realtime; default = 3.0)
%   save_video     - Boolean flag to save animation as MP4/AVI (default = false)

if nargin < 2 || isempty(P),              P = aircraftParameters(); end
if nargin < 3 || isempty(playback_speed), playback_speed = 3.0;     end
if nargin < 4 || isempty(save_video),     save_video = false;       end

time  = flight_log.time;
N     = length(time);
dt    = time(2) - time(1);

% Frame skipping based on playback speed to maintain smooth rendering
frame_skip = max(1, round(playback_speed * 0.05 / dt));

fig = figure('Color', 'k', 'Position', [100, 100, 1100, 600], 'Name', 'Flight Animation');
ax = axes(fig, 'Color', [0.05 0.08 0.12]);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
set(ax, 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], 'GridColor', [0.2 0.3 0.4]);

% 1. Static Geometry: Runway and Touchdown Markings
plot(ax, [-600, 1600], [0, 0], 'Color', [0.5 0.5 0.5], 'LineWidth', 4);
plot(ax, [0, 0], [-5, 25], 'Color', [0.2 0.9 0.2], 'LineWidth', 3); % Threshold
plot(ax, P.touchdown_aim_x, 0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', [1 0.8 0], 'LineWidth', 1.5);

% Approach Corridor & Nominal Glide-Slope Line
x_gs = linspace(-3200, P.touchdown_aim_x, 200);
h_gs = -(x_gs - P.touchdown_aim_x) * tan(-P.gamma_des);
plot(ax, x_gs, h_gs, 'w--', 'LineWidth', 1.2);

% Trajectory Trace Handle
h_trail = plot(ax, NaN, NaN, 'c-', 'LineWidth', 1.8);

% Aircraft Fuselage & Wing Handles
fuselage_scale = 35.0; % Scaled for visual visibility
h_fuselage = plot(ax, [NaN, NaN], [NaN, NaN], 'Color', [1 0.9 0.2], 'LineWidth', 4.0);
h_cg       = plot(ax, NaN, NaN, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% HUD Telemetry Text Box
hud_str = sprintf('TIME: %5.1f s\nALT : %5.1f m\nSPD : %5.1f m/s\nPITCH: %+4.1f deg\nSINK : %4.2f m/s\nELEV : %+4.1f deg\nTHR  : %3.0f %%', ...
                  0, flight_log.h(1), flight_log.V(1), rad2deg(flight_log.theta(1)), ...
                  flight_log.sink_rate(1), rad2deg(flight_log.delta_e(1)), flight_log.delta_T(1)*100);
h_hud = text(ax, 0.03, 0.92, hud_str, 'Units', 'normalized', 'Color', [0.2 1.0 0.4], ...
             'FontSize', 11, 'FontName', 'Consolas', 'FontWeight', 'bold', ...
             'BackgroundColor', [0 0 0 0.7], 'EdgeColor', [0.2 0.8 0.3]);

xlabel(ax, 'Longitudinal Distance x [m]', 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');
ylabel(ax, 'Altitude h [m]', 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');
title(ax, 'Autonomous Aircraft Landing Approach & Flare Visualization', ...
      'Color', 'w', 'FontSize', 13, 'FontWeight', 'bold');

% Setup Video Writer if requested
v_writer = [];
if save_video
    vid_path = fullfile(pwd, 'Results', 'Figures', 'flight_animation.mp4');
    try
        v_writer = VideoWriter(vid_path, 'MPEG-4');
        v_writer.FrameRate = 25;
        open(v_writer);
    catch
        v_writer = [];
    end
end

%% 2. Animation Loop
for k = 1:frame_skip:N
    if ~ishandle(fig), break; end
    
    xk = flight_log.x(k);
    hk = flight_log.h(k);
    thetak = flight_log.theta(k);
    
    % Update Trajectory Trail
    set(h_trail, 'XData', flight_log.x(1:k), 'YData', flight_log.h(1:k));
    
    % Update Pitch Attitude Orientation
    dx_f = (fuselage_scale / 2.0) * cos(thetak);
    dh_f = (fuselage_scale / 2.0) * sin(thetak);
    set(h_fuselage, 'XData', [xk - dx_f, xk + dx_f], 'YData', [hk - dh_f, hk + dh_f]);
    set(h_cg, 'XData', xk, 'YData', hk);
    
    % Dynamic Smooth Camera Tracking Window
    cam_span_x = 650;
    xlim(ax, [xk - cam_span_x * 0.75, xk + cam_span_x * 0.25]);
    ylim(ax, [-10, max(50, hk + 60)]);
    
    % Update Real-time HUD Telemetry
    hud_str = sprintf(['TIME : %5.1fs\n' ...
                       'DIST : %+6.1fm\n' ...
                       'ALT  : %5.1fm\n' ...
                       'SPD  : %5.1fm/s (%4.1fkt)\n' ...
                       'PITCH: %+5.1fdeg\n' ...
                       'PATH : %+5.1fdeg\n' ...
                       'AOA  : %+5.1fdeg\n' ...
                       'SINK : %5.2fm/s\n' ...
                       'ELEV : %+5.1fdeg\n' ...
                       'THR  : %3.0f%%'], ...
                      time(k), xk, hk, flight_log.V(k), flight_log.V(k)*1.94384, ...
                      rad2deg(thetak), rad2deg(flight_log.gamma(k)), ...
                      rad2deg(flight_log.alpha(k)), flight_log.sink_rate(k), ...
                      rad2deg(flight_log.delta_e(k)), flight_log.delta_T(k)*100);
    set(h_hud, 'String', hud_str);
    
    drawnow;
    
    if ~isempty(v_writer)
        frame = getframe(fig);
        writeVideo(v_writer, frame);
    end
end

% Touchdown Marker at Final Position
if ishandle(fig)
    plot(ax, flight_log.x(end), flight_log.h(end), 'rs', 'MarkerSize', 14, ...
         'MarkerFaceColor', 'r', 'LineWidth', 2);
    text(ax, flight_log.x(end), flight_log.h(end) + 15, ...
         sprintf('TOUCHDOWN: %s\nSink Rate = %.2f m/s\nSpeed = %.1f m/s', ...
                 flight_log.status, flight_log.sink_rate(end), flight_log.V(end)), ...
         'Color', 'y', 'FontSize', 11, 'FontWeight', 'bold', ...
         'BackgroundColor', [0 0 0 0.8]);
    drawnow;
end

if ~isempty(v_writer)
    close(v_writer);
    fprintf('[Animation] Video saved to: %s\n', vid_path);
end

end
