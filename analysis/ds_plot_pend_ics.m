function ds_plot_pend_ics(ax, pend, c, opt)
% DS_PLOT_PEND_ICS  Draw pendulum lines for each IC coloured by c value.
%   pend : (N,1) pendulum angles in rad (0=balanced up, ±pi=hanging down)
%   c    : (N,1) colour values in [0,1]
  arguments
    ax
    pend (:,1) double
    c    (:,1) double
    opt.L            = 1
    opt.alpha        = 0.3
    opt.linewidth    = 1
    opt.perfect_mask = []
  end

  N = numel(pend);
  if N == 0, return; end
  hold(ax, 'on');
  axis(ax, 'equal');

  if isempty(opt.perfect_mask)
    perfect = false(N, 1);
  else
    perfect = logical(opt.perfect_mask);
  end

  % All ICs as thin coloured lines (vectorised: 2×N matrix → N lines)
  L  = opt.L;
  X  = [zeros(1,N); L*sin(pend)'];
  Y  = [zeros(1,N); L*cos(pend)'];
  h  = plot(ax, X, Y, 'LineWidth', opt.linewidth);

  cmap_local = cool(256);
  c_idx = max(1, min(256, round(c * 255) + 1));
  for i = 1:N
    h(i).Color = [cmap_local(c_idx(i), :), opt.alpha];
  end

  % Perfect ICs overlaid in black
  if any(perfect)
    np = sum(perfect);
    Xp = [zeros(1,np); L*sin(pend(perfect))'];
    Yp = [zeros(1,np); L*cos(pend(perfect))'];
    plot(ax, Xp, Yp, 'k-', 'LineWidth', 1.5);
  end

  % Pivot marker
  plot(ax, 0, 0, 'k.', 'MarkerSize', 8);
  xlim(ax, [-L L] * 1.15);
  ylim(ax, [-L L] * 1.15);
  ax.XTick = [];
  ax.YTick = [];
end
