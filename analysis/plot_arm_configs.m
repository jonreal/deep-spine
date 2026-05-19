function plot_arm_configs(ax, q0, q1, c, opt)
% PLOT_ARM_CONFIGS Render 2-link arm configurations as colored patches.
  arguments
    ax
    q0 (:,1) double           % shoulder angles
    q1 (:,1) double           % elbow angles
    c  (:,1) double           % color values per configuration
    opt.L = 1                 % link length
    opt.alpha = 0.05          % edge alpha
    opt.linewidth = 2
    opt.show_shoulder = true
  end

  N = numel(q0);
  if N == 0, return; end

  L = opt.L;

  % Vectorized forward kinematics
  x1 = L * sin(q0);
  y1 = -L * cos(q0);
  x2 = x1 + L * sin(q0 + q1);
  y2 = y1 - L * cos(q0 + q1);

  % Build patch arrays: each column is [0; x1; x2; NaN] (4 x N)
  Xp = [zeros(1,N); x1'; x2'; nan(1,N)];
  Yp = [zeros(1,N); y1'; y2'; nan(1,N)];
  Cp = repmat(c', 4, 1);

  hold(ax, 'on');
  patch(ax, Xp, Yp, Cp, ...
    'FaceColor', 'none', 'EdgeColor', 'flat', ...
    'EdgeAlpha', opt.alpha, 'LineWidth', opt.linewidth);

  if opt.show_shoulder
    plot(ax, 0, 0, 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
  end

  xlim(ax, [-L L]);
  ylim(ax, [-2*L 0]);
  axis(ax, 'off');
end
