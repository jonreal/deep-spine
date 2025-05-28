function summary_table = analysis_pipeline(path_to_experiments,opt)
  arguments
    path_to_experiments
    opt.printfig (1,1) logical = false
    opt.outdir (1,1) string = 'analysis'
  end
  tic
  set(0, 'DefaultFigureVisible', 'on');

  % Create output directories
  savepath = [path_to_experiments,'/',char(opt.outdir)];
  if ~exist(savepath, 'dir')
      mkdir(savepath);
  end
  dashboard_path = [savepath,'/dashboard'];
  if ~exist(dashboard_path, 'dir')
      mkdir(dashboard_path);
  end
  summary_path = [savepath,'/summary'];
  if ~exist(summary_path, 'dir')
    mkdir(summary_path);
  end

  % Create synergy masks
  n = 6;
  mRf = diag(repmat([1,0],1,n/2));
  mRe = diag(repmat([0,1],1,n/2));
  mR = mRf + mRe;

  mSf = repmat([1,0; 0, 0],n/2,n/2) - mRf;
  mSe = repmat([0,0; 0, 1],n/2,n/2) - mRe;
  mS = mSf + mSe;

  mAf = repmat([0,0; 1, 0],n/2,n/2);
  mAe = repmat([0,1; 0, 0],n/2,n/2);
  mA = mAf + mAe;

  % Color map
  n = 256;
  blue = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
  red  = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
  cmap = [blue; red];

  % Main loop
  % Generate list of experments
  dirs = dir([path_to_experiments,'/*']);
  id = {};
  summary_table = table();
  tags_exsist = 0;

  for i = 1:length(dirs)
    name = dirs(i).name;
    if dirs(i).isdir && ~any(strcmp(name, {'.', '..', '.DS_Store'}))

      % Check if data file exists
      path = [path_to_experiments,'/',name];
      if exist([path,'/data.txt'], 'file')
          fprintf(['\nProcessing: ', path,'\n']);
      else
          fprintf(['\nSkipping: ', path,'\n']);
          continue;
      end

      % Extract data
      [episode, time, muscle_strength, Wi, We] = ...
          extract_data([path,'/data.txt']);

      % Extract parameters
      [keys_str, vals_str] = extract_parameters(path);
      keys = split(keys_str, newline);
      vals = split(vals_str, newline);

      % SVD Analysis
      [U, S, V] = cellfun(@(W) svd(W), We, 'UniformOutput', false);
      singular_vals = cellfun(@(s) diag(s), S, 'UniformOutput', false);
      singular_vals = [singular_vals{:}]';
      explained_var = [singular_vals'.^2 ./ sum(singular_vals'.^2)]';
      sum_of_var = cumsum(explained_var, 2);

      rank1_mats = cellfun( ...
        @(U, S, V) arrayfun( ...
          @(i) S(i,i) * U(:,i) * V(:,i)',1:size(S,1), ...
          'UniformOutput', false), ...
        U, S, V, 'UniformOutput', false);

      % Abolute synergy analysis
      %Rf_a = [cellfun(@(W) (sum((W .* mRf), 'all') ./ sum(mRf,'all')), We)]';
      %Re_a = [cellfun(@(W) (sum((W .* mRe), 'all') ./ sum(mRe,'all')), We)]';
      %Sf_a = [cellfun(@(W) (sum((W .* mSf), 'all') ./ sum(mSf,'all')), We)]';
      %Se_a = [cellfun(@(W) (sum((W .* mSe), 'all') ./ sum(mSe,'all')), We)]';
      %Af_a = [cellfun(@(W) (sum((W .* mAf), 'all') ./ sum(mAf,'all')), We)]';
      %Ae_a = [cellfun(@(W) (sum((W .* mAe), 'all') ./ sum(mAe,'all')), We)]';
      R_a = [cellfun(@(W) (sum((W .* mR), 'all') ./ sum(mR,'all')), We)]';
      S_a = [cellfun(@(W) (sum((W .* mS), 'all') ./ sum(mS,'all')), We)]';
      A_a = [cellfun(@(W) (sum((W .* mA), 'all') ./ sum(mA,'all')), We)]';

      % Relative synergy analysis
      %Rf_r = [cellfun(@(W) (sum((W .* mRf), 'all') ./ sum(W,'all')), We)]';
      %Re_r = [cellfun(@(W) (sum((W .* mRe), 'all') ./ sum(W,'all')), We)]';
      %Sf_r = [cellfun(@(W) (sum((W .* mSf), 'all') ./ sum(W,'all')), We)]';
      %Se_r = [cellfun(@(W) (sum((W .* mSe), 'all') ./ sum(W,'all')), We)]';
      %Af_r = [cellfun(@(W) (sum((W .* mAf), 'all') ./ sum(W,'all')), We)]';
      %Ae_r = [cellfun(@(W) (sum((W .* mAe), 'all') ./ sum(W,'all')), We)]';
      R_r = [cellfun(@(W) (sum((W .* mR), 'all') ./ sum(W,'all')), We)]';
      S_r = [cellfun(@(W) (sum((W .* mS), 'all') ./ sum(W,'all')), We)]';
      A_r = [cellfun(@(W) (sum((W .* mA), 'all') ./ sum(W,'all')), We)]';

      % Summary outcomes
      signals = { ...
        R_a, S_a, A_a, ...
        R_r, S_r, A_r, ...
      };
      signals_str = { ...
        'R_a', 'S_a', 'A_a', ...
        'R_r', 'S_r', 'A_r', ...
      };
      synergy_measures = {@max, @mean, @var};
      synergy_meausres_str = {'max', 'mean', 'var'};
      synergy_outcomes = cellfun( ...
        @(signal) cellfun( ...
          @(measure) measure(signal(end-4000:end)), ...
          synergy_measures, 'UniformOutput', false), ...
        signals, 'UniformOutput', false);

      row = [ ...
        {name}, ...
        num2cell(str2double(vals(:)')), ...
        synergy_outcomes{:}];

      summary_table = vertcat(summary_table, row);

      % Only create tags once
      if ~tags_exsist
        synergy_outcomes_str = cellfun( ...
          @(signal_str) cellfun( ...
            @(measure) [signal_str, '_', measure], ...
            synergy_meausres_str, 'UniformOutput', false), ...
          signals_str, 'UniformOutput',false);
        label = [ ...
          {'sim-ID'}, ...
          keys(:)', ...
          synergy_outcomes_str{:}
        ];
        tags_exsist = 1;
        summary_table.Properties.VariableNames = label;
      end

      % Figure
      figure('Position',[100 100 1120 420]);

        % W decompose legend
        xs = 0.31;
        w = 0.32;
        ys = 0.63;
        shift1 = 0.09;
        shift2 = 0.04;
        ax = axes('Position',[xs, ys, w, w]);
        p = imagesc( ...
          mRf ...
          + 0.70*mRe ...
          + 0.30*mSf ...
          + 0.10*mSe ...
          - 0.75*mAf ...
          - mAe);
        p.AlphaData = 0.8;
        axis square;
        axis off;
        caxis([-1 1]);

        legend_list = { {1, 'Flexion Reflex'}, ...
                        {0.7, 'Extension Reflex'}, ...
                        {0.3, 'Flexion Synergy'}, ...
                        {0.1, 'Extension Synergy'}, ...
                        {-0.75, 'Flexion Antagonist'}, ...
                        {-1, 'Extension Antagonist'} };

        ys = 0.56;
        space = 0.055;

        rgb_val = zeros(numel(legend_list),3);
        for i = 1:length(legend_list)
          ax = axes('Position',[xs + shift1, ys, 0.045, 0.045]);
          p = imagesc(legend_list{i}{1});
          axis square;
          yticklabels([])
          xticklabels([])
          caxis([-1 1]);

          c = legend_list{i}{1};
          rgb_map = colormap(ax);
          c_range = caxis(ax);
          idx = round((c - c_range(1)) / diff(c_range) * (size(rgb_map,1) - 1)) + 1;
          idx = max(1, min(idx, size(rgb_map,1)));  % clamp to valid range
          rgb_val(i,:) = rgb_map(idx, :);

          ann = annotation('textbox','Position',[xs+shift1+shift2 ys 0.5 0.045]);
          ann.String = legend_list{i}{2};
          ann.EdgeColor = 'none';
          ann.HorizontalAlignment = 'left';
          ys = ys - space;
        end

        % Time series traces
        xs = 0.05;
        ys = 0.60;
        w = 0.35;
        h = 0.22;
        pos = [xs, ys, w, h];
        ax = axes('Position',pos);
%        trace_list = { ...
%          {Rf_a, rgb_val(1,:)}, ...
%          {Re_a, rgb_val(2,:)}, ...
%          {Sf_a, rgb_val(3,:)}, ...
%          {Se_a, rgb_val(4,:)}, ...
%          {Af_a, rgb_val(5,:)}, ...
%          {Ae_a, rgb_val(6,:)}, ...
%        };
        trace_list = { ...
          {R_a, rgb_val(2,:)}, ...
          {S_a, rgb_val(4,:)}, ...
          {A_a, rgb_val(6,:)}, ...
        };
        for i=1:numel(trace_list)
          p = plot(trace_list{i}{1},'-o'); hold all;
          p.MarkerSize = 2;
          p.Color = trace_list{i}{2};
        end
        xticklabels({});
        axis tight;
        ylim([0,1]);
        ylabel({'Absolute','||W * M|| / ||M||'})
        ax.Color = 'none';

        ys = 0.35;
        pos = [xs, ys, w, h];
        ax = axes('Position',pos);
%        trace_list = { ...
%          {Rf_r, rgb_val(1,:)}, ...
%          {Re_r, rgb_val(2,:)}, ...
%          {Sf_r, rgb_val(3,:)}, ...
%          {Se_r, rgb_val(4,:)}, ...
%          {Af_r, rgb_val(5,:)}, ...
%          {Ae_r, rgb_val(6,:)}, ...
%        };
        trace_list = { ...
          {R_r, rgb_val(2,:)}, ...
          {S_r, rgb_val(4,:)}, ...
          {A_r, rgb_val(6,:)}, ...
        };
        for i=1:numel(trace_list)
          p = plot(trace_list{i}{1},'-o'); hold all;
          p.MarkerSize = 2;
          p.Color = trace_list{i}{2};
        end
        axis tight;
        xticklabels({});
        ylim([0,1]);
        ylabel({'Relative','||W * M|| / ||W||'})
        ax.Color = 'none';



        ys = 0.1;
        pos = [xs, ys, w, h];
        ax = axes('Position',pos);
        trace_list = { ...
          {explained_var(:,1), [1 0.1 0.1]}, ...
          {explained_var(:,2), [0.8 0.1 0.1]}, ...
          {explained_var(:,3), [0.6 0.1 0.1]}, ...
          {explained_var(:,4), [0.4 0.1 0.1]}, ...
          {explained_var(:,5), [0.2 0.1 0.1]}, ...
          {explained_var(:,6), [0 0.1 0.1]}, ...
        };
        for i=1:numel(trace_list)
          p = plot(trace_list{i}{1},'-o'); hold all;
          p.MarkerSize = 2;
          p.Color = trace_list{i}{2};
        end
        axis tight;
        ylim([0,1]);
        xlabel('Simulation Time (s)');
        ylabel('Explained Variance')
        ax.Color = 'none';

        ys = 0.85;
        h = 0.1;
        pos = [xs, ys, w, h];
        ax = axes('Position',pos);
        plot(muscle_strength,'-k');
        xticklabels({});
        axis tight;
        ylim([0,1]);
        ylabel('Strength')
        ax.Color = 'none';


        % SVD Analysis
        xs = 0.45;
        ys = 0.47;
        w = 0.45;
        h = 0.45;
        pos = [xs, ys, w, h];
        ax = axes('Position', pos);
          imagesc(We{1});
          axis square;
          ax = gca;
          ax.XAxisLocation = 'top';
          yticks(1:6);
          yticklabels({'MN-SF', 'MN-SE', 'MN-EF', 'MN-EE', 'MN-BF', 'MN-BE'});
          xticklabels({'Ia-SF', 'Ia-SE', 'Ia-EF', 'Ia-EE', 'Ia-BF', 'Ia-BE'});
          colormap(ax,cmap);
          caxis([0 1]);

          label_pos = pos .* [1 0.925 1 0.1];
          ann = annotation('textbox','Position',label_pos);
          ann.String = 'Initial';
          ann.EdgeColor = 'none';
          ann.HorizontalAlignment = 'center';

      xshift = 0.2;
      pos(1) = pos(1) + xshift;
      ax = axes('Position',pos);
        imagesc(We{end});
        axis square;
        ax = gca;
        ax.XAxisLocation = 'top';
        yticklabels([])
        xticklabels([])
        colormap(ax,cmap);
        caxis([0 1]);
        colorbar;

        label_pos = pos .* [1 0.925 1 0.1];
        ann = annotation('textbox','Position',label_pos);
        ann.String = 'Final';
        ann.EdgeColor = 'none';
        ann.HorizontalAlignment = 'center';

        xs = 0.52;
        w = 0.15;
        sp = 0.015;
        for j=1:2
          for i=1:3
            pos = [xs + (w*0.45)*(i-1), 0.25 - (w + sp)*(j-1) , w, w];
            ax = axes('Position',pos);
            imagesc(rank1_mats{end}{i + 3*(j-1)});
              axis square;
              ax = gca;
              ax.XAxisLocation = 'top';
              yticklabels([])
              xticklabels([])
              colormap(ax,cmap);
              caxis([-1 1]);
          end
        end

        pos = [xs + 0.05, 0.025, w, w];
        ann = annotation('textbox','Position',pos);
        ann.String = 'Rank 1 matrices';
        ann.EdgeColor = 'none';
        ann.HorizontalAlignment = 'center';
        ann.VerticalAlignment = 'bottom';

        xs = 0.8;
        ys = 0.05;
        w = 0.15;
        h = 0.35;
        pos = [xs, ys, w, h];
        ax = axes('Position',pos);
          bar(explained_var(end,:), 'k', 'LineWidth', 2);
          hold all;
          plot(sum_of_var(end,:), '-or', 'LineWidth', 2);
          ylabel('Explained Variance');
          ylim([0,1]);
        ax.Color = 'none';

        xs = 0.41;
        ann = annotation('textbox','Position',[xs,0,0.5,0.25]);
        ann.Interpreter = 'none';
        ann.String = keys_str;
        ann.EdgeColor = 'none';
        ann.HorizontalAlignment = 'left';
        ann.VerticalAlignment = 'top';
        ann.FontSize = 7;

        ann = annotation('textbox','Position',[xs,0,0.12,0.25]);
        ann.Interpreter = 'none';
        ann.String = vals_str;
        ann.EdgeColor = 'none';
        ann.HorizontalAlignment = 'right';
        ann.VerticalAlignment = 'top';
        ann.FontSize = 7;

        para_str = cellfun(@(v) ['_',v],vals,'UniformOutput',false);
        sim_fig_path = [ ...
          dashboard_path, ...
          '/', name, ...
          para_str{:}, ...
          '.png'];
        print(sim_fig_path,'-dpng');
        fprintf(['Saved figure to: ', sim_fig_path, '\n\n']);
        close(gcf);
    end
  end
  save([summary_path,'/summary_table.mat'],'summary_table');
  %exp_summary(summary_table,outdir=[savepath,'/summary']);
  toc
end
