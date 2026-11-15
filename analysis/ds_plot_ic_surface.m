function hull_area = ds_plot_ic_surface(ax, q0, q1, rew, opt)
% DS_PLOT_IC_SURFACE Reward-weighted KDE surface + perfect-IC scatter + convex hull.
  arguments
    ax
    q0 (:,1) double
    q1 (:,1) double
    rew (:,1) double
    opt.perfect_mask = []     % logical per-episode perfect flag
    opt.rew_max = NaN         % for cross-group normalization
    opt.do_kernel = true
    opt.grid_res = 50
    opt.markersize = 20
    opt.view_angle = [0, 90]
    opt.q0_lim = [NaN NaN]
    opt.q1_lim = [NaN NaN]
  end

  hull_area = 0;
  hold(ax, 'on');

  if any(isnan(opt.q0_lim)), opt.q0_lim = [min(q0), max(q0)]; end
  if any(isnan(opt.q1_lim)), opt.q1_lim = [min(q1), max(q1)]; end
  if isnan(opt.rew_max), opt.rew_max = max(rew); end

  if ~isempty(opt.perfect_mask)
    g_success = logical(opt.perfect_mask);
  else
    g_success = false(size(rew));
  end

  % KDE surface weighted by reward — high-reward ICs contribute more
  if opt.do_kernel
    rew_norm = rew / opt.rew_max;
    w = max(0, rew_norm);           % non-negative weights

    [X1, X2] = meshgrid(linspace(opt.q0_lim(1), opt.q0_lim(2), opt.grid_res), ...
                         linspace(opt.q1_lim(1), opt.q1_lim(2), opt.grid_res));
    grid_pts = [X1(:), X2(:)];

    if sum(w) > 0
      Z = reshape(ksdensity([q0, q1], grid_pts, 'Weights', w), size(X1));
      Z = Z / max(Z(:));
    else
      Z = zeros(size(X1));
    end

    rgb_gray = repmat(Z, 1, 1, 3);
    surf(ax, X1, X2, Z + 0.01, rgb_gray, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    view(ax, opt.view_angle);
    zlim(ax, [0 1.1]);
  end

  % Scatter: perfect ICs only at z=1
  if sum(g_success) > 0
    scatter3(ax, q0(g_success), q1(g_success), ones(sum(g_success),1), ...
      opt.markersize, 'k', 'filled', 'Marker', 'o');
  end

  % Convex hull of perfect ICs
  if sum(g_success) >= 3
    pq0 = q0(g_success);
    pq1 = q1(g_success);
    try
      [k, hull_area] = convhull(pq0, pq1);
      plot3(ax, pq0(k), pq1(k), ones(size(k)), 'r-', 'LineWidth', 2);
    catch
    end
  end

  xlim(ax, opt.q0_lim);
  ylim(ax, opt.q1_lim);
end
