function exp_summary(tab,opt)
  arguments
    tab
    opt.outdir (1,1) string = './summary'
  end

  muscle_strengths = unique(tab.opt_max_mus_str);
  babble_prob = unique(tab.sup_kwa_bab_max_pro);
  gains = unique(tab.('heb_kwa_{2}_gai'));

  cmap = cbrewer('qual','Accent',numel(gains));

  alpha = 0.5
  N = 100;

  quiries = { ...
    'R_r_mean', 'S_r_mean', 'R_a_mean', 'S_a_mean', 'A_a_mean', 'A_r_mean' ...
  };
  for q=1:numel(quiries)
    figure;
    legend_list = [];
    for i=1:numel(gains)

      iq = find(tab.('heb_kwa_{2}_gai') == gains(i));
      x = tab.opt_max_mus_str(iq);
      y = tab.sup_kwa_bab_max_pro(iq);
      z = tab.(quiries{q})(iq);

      p = scatter3(x,y,z)
      p.MarkerEdgeColor = 'w'
      p.MarkerFaceColor = cmap(i,:);

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
      p.FaceColor = cmap(i,:);
      p.FaceAlpha = alpha;
      p.EdgeColor = cmap(i,:);
      legend_list(end+1) = p;

      xlabel('Muscle Strength');
      ylabel('Babble Probability')
      p = title(quiries{q})
      p.Interpreter = 'none';

    end
    gains_str = arrayfun(@num2str, gains, 'UniformOutput', false);
    legend(legend_list,gains_str)
    if ~exist(opt.outdir, 'dir')
        mkdir(opt.outdir);
    end
    print([opt.outdir{:},'/',quiries{q},'.png'],'-dpng');
    fprintf(['Saved figure to: ', [opt.outdir{:},'/',quiries{q},'.png'], '\n\n']);
    %close(gcf);
  end
end
