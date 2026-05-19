function plot_heatmap(ax,x,y,z,opt)
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
    opt.MarkerSize = 5;
    opt.cmap = flipud(brewermap(1024, 'RdBu'));
  end

  %n = 1024;
  %blue = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
  %red  = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
  %cmap = [blue; red];

  p = scatter3(x,y,z,5,'filled');
  p.MarkerFaceColor = 'k';
  p.MarkerFaceAlpha = 0.8;

  % Create interpolant object
  F = scatteredInterpolant(x, y, z, 'linear', 'none');

  % Create regular grid
  [X, Y] = meshgrid(linspace(min(x), max(x), opt.N), ...
                    linspace(min(y), max(y), opt.N));

  % Evaluate interpolant on grid
  Z = F(X, Y);

  % Plot with mesh or surf
  hold all;
  p = surf(X, Y, Z);
  p.FaceAlpha = opt.alpha;
  p.EdgeColor = 'none';
  %legend_list(end+1) = p;

  %imagesc([min(X),max(X)],[min(Y),max(Y)],Z);
  %[~,p] = contourf(X, Y, Z);
  %p.FaceAlpha = alpha;
  colormap(opt.cmap);
  caxis(opt.zl);

  ax.YAxis.Scale = opt.scale;
  xlim(opt.xl);
  ylim(opt.yl);
  zlim(opt.zl);
  view(0,90);


  ax.Color = 'none';

  if ~opt.XTickLabels
    ax.XAxis.TickLabels = {};
  end
  if ~opt.YTickLabels
    ax.YAxis.TickLabels = {};
  end

  axis square;

end
