function Z_all = exp_summary(tab,opt)
  arguments
    tab
    opt.outdir (1,1) string = './summary'
  end

  muscle_strengths = unique(tab.opt_max_mus_str);
  babble_prob = unique(tab.sup_kwa_bab_max_pro);
  gains = unique(tab.('heb_kwa_gai'));

  %cmap = flipud(cbrewer('div','RdBu',64));
  %
  cmap = flipud(brewermap(1024, 'RdBu'));

  %n = 64;
  %blue = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
  %red  = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
  %cmap = [blue; red];

  %cmap = flipud(gray);

  alpha = 1;
  N = 1000;
  scale = 'log';


  nrows = numel(gains);
  ncols = 1;

  pxh = 800;
  pxw = pxh;

  figure('Position', [431 28 958 1309]);
  xl = [min(tab.opt_max_mus_str), max(tab.opt_max_mus_str)];
  yl = [min(tab.sup_kwa_bab_max_pro), max(tab.sup_kwa_bab_max_pro)];

  xl = [0.1 1];
  yl = [0.0003    0.0100];
      zl = [0 1];


  ysp = 0.02;
  xsp = 0.02;
  xpad = 0.07;
  ypad = 0.07;

  gainFudge = 0.9;

  h = (1 - 2*ypad - (nrows-1)*ysp) / nrows;
  w = (1 - 2*xpad - (ncols-1)*xsp) / ncols;

  x0 = xpad;
  y0 = 1 - ypad - h;

  legend_list = [];
  axes_list = zeros(1, nrows*ncols);
  indx = 1;
  ys = y0;
  xs = x0;

  %Z_all = cell(numel(gains), numel(quiries));
  for i=1:numel(gains)
    pos = [xs, ys, w, h];

    if i==1
      ann = annotation('textbox',pos + [-xpad*gainFudge, ypad/2, -0.1, 0]);
      ann.String = 'Gain';
      ann.FontSize = 12;
      ann.HorizontalAlignment = 'left';
      ann.VerticalAlignment = 'top';
      ann.EdgeColor = 'none';
    end

      ax = axes('Position', pos);

      iq = find(tab.('heb_kwa_gai') == gains(i));
      x = tab.opt_max_mus_str(iq);
      y = tab.sup_kwa_bab_max_pro(iq);
      z = tab.score_mean(iq);

      plot_heatmap(ax,x,y,z);


      return

      p = scatter3(x,y,z);
      p.MarkerEdgeColor = cmap(i,:);
      p.MarkerFaceColor = cmap(i,:);
      p.Marker = '.';

      % Create interpolant object
      F = scatteredInterpolant(x, y, z, 'linear', 'none');

      % Create regular grid
      [X, Y] = meshgrid(linspace(min(x), max(x), N), ...
                        linspace(min(y), max(y), N));

      % Evaluate interpolant on grid
      Z = F(X, Y);

      hold all;
      % Plot with mesh or surf
      p = surf(X, Y, Z);
      p.FaceAlpha = alpha;
      p.EdgeColor = 'none';
      legend_list(end+1) = p;

      %imagesc([min(X),max(X)],[min(Y),max(Y)],Z);
      %[~,p] = contourf(X, Y, Z);
      %p.FaceAlpha = alpha;
      colormap(cmap);
      caxis(zl);

      ax.YAxis.Scale = scale;

      if ~(i == numel(gains))
        ax.XAxis.TickLabels = {};
      end

      if i==numel(gains)
        xlabel('Strength');
        ylabel('Babble Probability');
      end



      axes_list(indx) = ax;
      indx = indx + 1;


      xlim(xl);
      ylim(yl);
      zlim(zl);

      view(0,90)

    %gains_str = arrayfun(@num2str, gains, 'UniformOutput', false);
    %legend(legend_list,gains_str)
    %if ~exist(opt.outdir, 'dir')
    %    mkdir(opt.outdir);
    %end
    %print([opt.outdir{:},'/',quiries{q},'.png'],'-dpng');
    %ax = gca;
    %%close(gcf);
    ys = ys - h - ysp;
  end
end
%  linkaxes(axes_list,'xyz');
%  %fprintf(['Saved figure to: ', [opt.outdir{:},'/summary.png'], '\n\n']);
%  %
%  p = colorbar;
%  p.Position = [1-xpad*0.9, ypad, xpad/4, 1-2*ypad];
%
%  xs = x0 + (ncols-1)*(w + xspace);
%  ys = y0;
%  for g=1:numel(gains)
%    pos = [xs, ys, w, h];
%
%    if g==1
%      ann = annotation('textbox',pos + [0, ypad/2, 0, 0]);
%      ann.String = qlab{end};
%      ann.FontSize = 12;
%      ann.HorizontalAlignment = 'center';
%      ann.VerticalAlignment = 'top';
%      ann.EdgeColor = 'none';
%    end
%
%    ax = axes('Position', pos);
%    hold all;
%    score = Z_all{g,1}.*Z_all{g,2}.*(1 - Z_all{g,3});
%
%    %size(score)
%    
%
%    [max_score,indx] = max(score(:));
%    xm = X(:);
%    xm = xm(indx);
%    ym = Y(:);
%    ym = ym(indx);
%
%    %size(x)
%    %size(X)
%    %INDX = arrayfun(@(i) find(x(i) == X(:)),1:numel(x),'UniformOutput',0);
%
%    %return
%
%    p = surf(X, Y, score);
%    p.FaceAlpha = alpha;
%    p.EdgeColor = 'none';
%    legend_list(end+1) = p;
%
%
%    %size(x)
%    %size(y)
%    %size(score)
%    %p = scatter3(x,y,score);
%
%    %p.MarkerEdgeColor = cmap(i,:);
%    %p.MarkerFaceColor = cmap(i,:);
%    %p.Marker = '.';
%
%    %ps = scatter3(xm,ym,max_score);
%    %ps.MarkerFaceColor = 'black';
%    %ps.MarkerEdgeColor = 'black';
%    %text(xm+0.02,ym,max_score,num2str(max_score));
%
%
%    %imagesc([min(X),max(X)],[min(Y),max(Y)],Z);
%    %[~,p] = contourf(X, Y, score);
%    %p.FaceAlpha = alpha;
%    colormap(cmap);
%    caxis(zl);
%
%    ax.YAxis.Scale = scale;
%
%    xlim(xl);
%    ylim(yl);
%    zlim(zl);
%
%    view(0,90)
%
%    ax.YAxis.TickLabels = {};
%    if ~(g == numel(gains))
%      ax.XAxis.TickLabels = {}
%    end
%
%
