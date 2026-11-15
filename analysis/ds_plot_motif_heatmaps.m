function ds_plot_motif_heatmaps(tab, opt)
%DS_PLOT_MOTIF_HEATMAPS  Sweep heatmaps of the motif metrics vs (babble, strength).
%   One ROW per Hebbian gain (kappa); columns = mean Reflex (R), Synergy (S),
%   Antagonist (A), and selectivity S-A. Each panel interpolates the metric
%   over the (p_b, lambda_max) grid via ds_plot_heatmap, with babble probability
%   p_b (the dominant driver) on x and strength lambda_max on y.
%
%   ds_plot_motif_heatmaps(summary_table)
%   ds_plot_motif_heatmaps(summary_table, cmap=turbo(1024))
  arguments
    tab
    opt.cmap = turbo(1024);
    opt.selzl = [0 0.30];        % selectivity range; S-A spans ~[-0.01, 0.30]
    opt.selcmap = [];            % [] -> built-in white->teal ramp
    opt.selectivity (1,1) logical = true   % false: drop the S-A column and its colorbar
  end
  if isempty(opt.selcmap)
    n = 1024; t = linspace(0,1,n)';
    opt.selcmap = [1-0.90*t, 1-0.55*t, 1-0.35*t];   % white -> deep teal
  end

  gains = unique(tab.('heb_kwa_gai'));
  nrows = numel(gains);
  ncols = 3 + opt.selectivity;

  zfields = {'r_mean','s_mean','a_mean','sel_mean'};
  ctitles = {'Reflex (R)','Synergy (S)','Antagonist (A)','Selectivity (S - A)'};

  % ds_plot_heatmap log-scales y only, so p_b is passed as log10(p_b) on a linear
  % x-axis (interpolation then happens in log space) with 10^k tick labels.
  xl = log10([0.001 0.1]);
  yl = [0.1 1];
  zl = [0 1];
  % Selectivity is a signed difference on a much smaller scale than R/S/A, so
  % it gets its own limits and colormap; sharing [0,1] would render it blank.
  zls   = {zl, zl, zl, opt.selzl};
  cmaps = {opt.cmap, opt.cmap, opt.cmap, opt.selcmap};
  zfields = zfields(1:ncols);  ctitles = ctitles(1:ncols);
  zls     = zls(1:ncols);      cmaps   = cmaps(1:ncols);

  % ---- layout: gain-label column | 4-heatmap grid | right colorbar ----
  figH = round(200*nrows + 140);
  figure('Position', [431 60 1100 figH]);

  ysp   = 0.02;      % gap between rows
  xsp   = 0.02;      % gap between heatmaps
  xpad  = 0.03;      % left figure margin
  wlabs = 0.06;      % gain-label column width
  ypad  = 0.07;
  cbw   = 0.015;     % right colorbar width
  cgap  = 0.02;

  h = (1 - 2*ypad - (nrows-1)*ysp) / nrows;
  x_hm0      = xpad + wlabs;
  x_hm_right = 1 - xpad - cbw - cgap;
  w = (x_hm_right - x_hm0 - (ncols-1)*xsp) / ncols;

  ys = 1 - ypad - h;
  for i = 1:nrows

    iq = find(tab.('heb_kwa_gai') == gains(i));
    x  = log10(tab.sup_kwa_bab_max_pro(iq));
    y  = tab.opt_max_mus_str(iq);

    xs = x_hm0;
    for c = 1:ncols
      pos = [xs, ys, w, h];
      ax  = axes('Position', pos);
      z   = tab.(zfields{c})(iq);

      xtl = (i==nrows) && (c==1);   % tick labels only on the bottom-left panel
      ytl = (i==nrows) && (c==1);
      ds_plot_heatmap(ax,x,y,z, xl=xl,yl=yl,zl=zls{c},scale='linear', ...
                   XTick=[-3 -2 -1], YTick=linspace(0.1,1,4), ...
                   XTickLabel={'10^{-3}','10^{-2}','10^{-1}'}, ...
                   XTickLabels=xtl, YTickLabels=ytl, cmap=cmaps{c});

      if (i==nrows) && (c==1)
        xlabel(ax,'Babble Probability');
        ylabel(ax,'Strength');
      end

      if i==1                        % column titles on the top row
        ann = annotation('textbox','Position',pos+[0,ypad/2,0,0]);
        ann.String = ctitles{c};
        ann.FontSize = 12;
        ann.EdgeColor = 'none';
        ann.HorizontalAlignment = 'center';
        ann.VerticalAlignment = 'top';
      end

      if c==1                        % row label = Hebbian gain (kappa)
        ann = annotation('textbox','Position',[0, ys, x_hm0, h]);
        ann.String = sprintf('\\kappa = %g', gains(i));
        ann.FontSize = 12;
        ann.EdgeColor = 'none';
        ann.HorizontalAlignment = 'left';
        ann.VerticalAlignment = 'middle';
      end

      xs = xs + w + xsp;
    end
    ys = ys - h - ysp;
  end

  % ---- two colorbars: R/S/A share [0,1]; selectivity has its own ----
  Hc = (1 - 2*ypad - 0.06)/2;                 % each bar, with a gap between
  draw_cbar([x_hm_right + cgap, ypad + Hc + 0.06, cbw, Hc], opt.cmap,    zl,        'R, S, A');
  if opt.selectivity
    draw_cbar([x_hm_right + cgap, ypad,           cbw, Hc], opt.selcmap, opt.selzl, 'S - A');
  end
end

function draw_cbar(pos, cmap, lims, lbl)
  ax  = axes('Position', pos);
  pcb = imagesc([0 1],[0 1], flipud(linspace(0,1,256)'));
  pcb.AlphaData = 0.8;
  colormap(ax, cmap);  caxis(ax,[0 1]);
  ylim(ax,[0 1]);  ax.XAxis.Visible = 'off';
  ax.YAxisLocation = 'right';
  yticks(ax,[0 1]);  yticklabels(ax, {num2str(lims(2)), num2str(lims(1))});
  title(ax, lbl, 'FontSize', 9, 'FontWeight','normal');
end
