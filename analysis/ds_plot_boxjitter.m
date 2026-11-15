function ds_plot_boxjitter(ax, groups, opt)
% DS_PLOT_BOXJITTER Boxplot with deterministic jittered scatter overlay.
  arguments
    ax
    groups cell               % cell array of numeric vectors, one per group
    opt.colors = []           % [n_groups x 3]
    opt.markersize = 40
    opt.jitter_width = 0.15
    opt.box_width = 0.1
    opt.labels = {}
  end

  n_groups = numel(groups);
  if isempty(opt.colors)
    opt.colors = lines(n_groups);
  end

  hold(ax, 'on');

  % Build flat arrays for boxplot
  boxplot_data = [];
  boxplot_groups = [];
  for g = 1:n_groups
    vals = groups{g}(:);
    n = numel(vals);

    boxplot_data = [boxplot_data; vals];
    boxplot_groups = [boxplot_groups; g*ones(n,1)];

    % Deterministic jitter: evenly spaced
    jitter = opt.jitter_width * (1:n) / n;

    % Two-group special case: face jitter inward
    if n_groups == 2
      if g == 1
        xshift = 0.35;
      else
        xshift = -0.35;
      end
    else
      xshift = 0.35;
    end

    scatter(ax, g + xshift + jitter, vals, opt.markersize, ...
      opt.colors(g,:), 'filled', 'MarkerFaceAlpha', 0.5);
  end

  boxplot(ax, boxplot_data, boxplot_groups, ...
    'Widths', opt.box_width, 'Colors', opt.colors);
  xticks(ax, 1:n_groups);
  xlim(ax, [0.5, n_groups + 0.7]);

  if ~isempty(opt.labels)
    xticklabels(ax, opt.labels);
  else
    xticklabels(ax, 1:n_groups);
  end
end
