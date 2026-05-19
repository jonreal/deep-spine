logBase = '/Users/jonreal/repos/deep-spine/experiments/hebbian/logs/';
exp1 = 'exp8/sweep_strength_babble_gain_0/';
file = 'analysis/summary/summary_table.mat';

S = load([logBase,exp1,file])

%exp_summary(S.summary_table)
%
plot_heatmaps({S.summary_table,S.summary_table},3)


function plot_heatmaps(tabs,gain)

  ntabs = numel(tabs);

  tab = tabs{1};
  cmap = flipud(cbrewer('div','RdBu',64));
  alpha = 1;
  N = 500;
  scale = 'log';
  quiries = { ...
    'R_a_mean', 'S_a_mean', 'A_a_mean' ...
  };
  nrows = 1;
  ncols = numel(quiries) + 1;

  aspect_ratio = nrows / ncols;
  pxh = 800;
  pxw = pxh / aspect_ratio;
  figure('Position', [431 28 958 1309]);


  muscle_strengths = unique(tab.opt_max_mus_str);
  babble_prob = unique(tab.sup_kwa_bab_max_pro);
  gains = unique(tab.('heb_kwa_{2}_gai'));

 
  %xl = [min(tab.opt_max_mus_str), max(tab.opt_max_mus_str)];
  %yl = [min(tab.sup_kwa_bab_max_pro), max(tab.sup_kwa_bab_max_pro)];

  xl = [0.2 0.6];
  yl = [0.0003    0.0100];

  zl = [0, 1];

  yspace = 0.01;
  xspace = 0.02;
  xpad = 0.07;
  ypad = 0.045;

  gainFudge = 0.9;

  h = ((1 - 2*ypad - (nrows-1)*yspace) / nrows) / 3;
  w = (1 - 2*xpad - (ncols-1)*xspace) / ncols;

  x0 = xpad;
  y0 = 1 - ypad - h;

  legend_list = [];
  axes_list = zeros(1, nrows*ncols);
  indx = 1;
  ys = y0;

  qlab = {'Reflex (R)', ...
          'Synergy (S)', ...
          'Anti-synergy (A)' ...
           'R*S*(1-A)'};

  Z_all = cell(1, numel(quiries));

  xs = x0;
  i = gain;
  for q=1:numel(quiries)

    pos = [xs, ys, w, h];

    if i==1
      ann = annotation('textbox',pos + [0, ypad/2, 0, 0]);
      ann.String = qlab{q};
      ann.FontSize = 12;
      ann.HorizontalAlignment = 'center';
      ann.VerticalAlignment = 'top';
      ann.EdgeColor = 'none';

      if q==1
        ann = annotation('textbox',pos + [-xpad*gainFudge, ypad/2, 0, 0]);
        ann.String = 'Gain';
        ann.FontSize = 12;
        ann.HorizontalAlignment = 'left';
        ann.VerticalAlignment = 'top';
        ann.EdgeColor = 'none';
      end
    end

    ax = axes('Position', pos);

    iq = find(tab.('heb_kwa_{2}_gai') == gains(i));
    x = tab.opt_max_mus_str(iq);
    y = tab.sup_kwa_bab_max_pro(iq);
    z = tab.(quiries{q})(iq);

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
    Z_all{q} = Z;

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

    if ~(q == 1)
      ax.YAxis.TickLabels = {};
    end
    if ~(i == numel(gains))
      ax.XAxis.TickLabels = {};
    end

    if q==1 && i==numel(gains)
      xlabel('Strength');
      ylabel('Babble Probability');
    end



    axes_list(indx) = ax;
    indx = indx + 1;


    xlim(xl);
    ylim(yl);
    zlim(zl);

    view(0,90)

    if q ==1
      ann = annotation('textbox',pos + [-xpad*gainFudge, 0, 0, 0]);
      ann.String = num2str(gains(i));
      ann.FontSize = 12;
      ann.HorizontalAlignment = 'left';
      ann.VerticalAlignment = 'middle';
      ann.EdgeColor = 'none';
    end

    xs = xs + w + xspace;
  end
  ys = ys - (h + yspace);
  %gains_str = arrayfun(@num2str, gains, 'UniformOutput', false);
  %legend(legend_list,gains_str)
  %if ~exist(opt.outdir, 'dir')
  %    mkdir(opt.outdir);
  %end
  %print([opt.outdir{:},'/',quiries{q},'.png'],'-dpng');
  %ax = gca;
  %%close(gcf);
  %
  p = colorbar;
  p.Position = [1-xpad*0.9, ypad, xpad/4, 1-2*ypad];

  xs = x0 + (ncols-1)*(w + xspace);
  ys = y0;
  g = gain;

  pos = [xs, ys, w, h];
  ax = axes('Position', pos);
  hold all;
  score = Z_all{1}.*Z_all{2}.*(1 - Z_all{3});

  [max_score,indx] = max(score(:));
  xm = X(:);
  xm = xm(indx);
  ym = Y(:);
  ym = ym(indx);

  p = surf(X, Y, score);
  p.FaceAlpha = alpha;
  p.EdgeColor = 'none';
  legend_list(end+1) = p;

    %ps = scatter3(xm,ym,max_score);
    %ps.MarkerFaceColor = 'black';
    %ps.MarkerEdgeColor = 'black';
    %text(xm+0.02,ym,max_score,num2str(max_score));


    %imagesc([min(X),max(X)],[min(Y),max(Y)],Z);
    %[~,p] = contourf(X, Y, score);
    %p.FaceAlpha = alpha;
    colormap(cmap);
    caxis(zl);

    ax.YAxis.Scale = scale;

    xlim(xl);
    ylim(yl);
    zlim(zl);

    view(0,90)





end
