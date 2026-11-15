function ds_plot_heatmap(ax,x,y,z,opt)
  arguments
    ax
    x
    y
    z
    opt.xl = [0.1,1];
    opt.yl = [0.0003 0.0100];
    opt.zl = [0 1];
    opt.scale = 'log';
    opt.alpha = 0.8;
    opt.N = 1000;
    opt.XTickLabels = false
    opt.YTickLabels = false
    opt.XTick = []           % uniform tick locations (empty = auto)
    opt.YTick = []
    opt.XTickLabel = {}      % custom label strings shown when *TickLabels true
    opt.YTickLabel = {}
    opt.TickDir = 'in'       % 'out' makes ticks visible outside the surf face
    opt.MarkerSize = 5;
    opt.cmap = parula(1024);
  end

  %n = 1024;
  %blue = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
  %red  = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
  %cmap = [blue; red];

  p = scatter3(x,y,z,5,'filled');
  p.MarkerFaceColor = 'k';
  p.MarkerFaceAlpha = 0.8;

  % Interpolate in the space the axis is DRAWN in. opt.scale applies to the
  % y-axis only (x stays linear), so when it is 'log' both the interpolant and
  % the query grid work in log10(y):
  %
  %   * grid: linspace in linear y gives ~91 rows to the decade 1e-3..1e-2 and
  %     ~909 to 1e-2..1e-1, yet each decade occupies half the panel -- the lower
  %     half renders at a tenth of the resolution. Spacing in log10 splits them
  %     evenly (500/500).
  %   * interpolant: built on raw y it ramps linearly in linear-y between
  %     log-spaced samples, which is not what a log axis displays. Small in
  %     magnitude (~0.03 on a 0..1 range) but free to fix here.
  ylog = strcmpi(opt.scale, 'log');
  if ylog && min(y) <= 0
    error('ds_plot_heatmap:nonPositiveY', ...
      ['scale=''log'' needs y > 0, but min(y) = %g. Check the swept ' ...
       'parameter or pass scale=''linear''.'], min(y));
  end

  if ylog
    F  = scatteredInterpolant(x, log10(y), z, 'linear', 'none');
    yq = linspace(log10(min(y)), log10(max(y)), opt.N);
  else
    F  = scatteredInterpolant(x, y, z, 'linear', 'none');
    yq = linspace(min(y), max(y), opt.N);
  end

  [X, Yq] = meshgrid(linspace(min(x), max(x), opt.N), yq);

  % Evaluate interpolant on grid
  Z = F(X, Yq);

  % back to data coordinates for surf / the log-scaled axis
  if ylog, Y = 10.^Yq; else, Y = Yq; end

  % Plot with mesh or surf
  hold all;
  p = surf(X, Y, Z);
  p.FaceAlpha = opt.alpha;
  p.EdgeColor = 'none';
  %legend_list(end+1) = p;

  %imagesc([min(X),max(X)],[min(Y),max(Y)],Z);
  %[~,p] = contourf(X, Y, Z);
  %p.FaceAlpha = alpha;
  colormap(ax,opt.cmap);
  caxis(opt.zl);

  ax.YAxis.Scale = opt.scale;
  xlim(opt.xl);
  ylim(opt.yl);
  zlim(opt.zl);
  view(0,90);


  ax.Color = 'none';
  ax.TickDir = opt.TickDir;            % uniform tick marks on every heatmap

  % uniform tick locations (applied to every heatmap when provided)
  if ~isempty(opt.XTick), xticks(ax, opt.XTick); end
  if ~isempty(opt.YTick), yticks(ax, opt.YTick); end

  % tick labels: shown only where requested; use custom strings if given
  if opt.XTickLabels
    if ~isempty(opt.XTickLabel)
      xticklabels(ax, opt.XTickLabel);
      ax.XAxis.TickLabelInterpreter = 'tex';
    end
  else
    ax.XAxis.TickLabels = {};
  end
  if opt.YTickLabels
    if ~isempty(opt.YTickLabel)
      yticklabels(ax, opt.YTickLabel);
      ax.YAxis.TickLabelInterpreter = 'tex';
    end
  else
    ax.YAxis.TickLabels = {};
  end

  axis square;

end
