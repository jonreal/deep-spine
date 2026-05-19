function summary_table = analysis_pipeline(path_to_experiments,opt)
  arguments
    path_to_experiments
    opt.printfig (1,1) logical = false
    opt.outdir (1,1) string = 'analysis-2'
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
  n = 1024;
  blue = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
  red  = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
  cmap = [blue; red];

  %cmap = cbrewer('div', 'RdBu', n);
  %cmap = flipud(cmap);   % low = blue, high = red
  %cmap = flipud(brewermap(n,'RdYlBu'));

  %cmap = flipud(cbrewer('div','RdGy',64));

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
      R_a = [cellfun(@(W) (sum((W .* mR), 'all') ./ sum(mR,'all')), We)]';
      S_a = [cellfun(@(W) (sum((W .* mS), 'all') ./ sum(mS,'all')), We)]';
      A_a = [cellfun(@(W) (sum((W .* mA), 'all') ./ sum(mA,'all')), We)]';

      % Relative synergy analysis
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
          @(measure) measure(signal(end-100:end)), ...
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
      figure('Position',[100 100 239   175]);

        xs = 0.01;
        ys = 0.5;
        w = 0.45;
        h = 0.45;
        pos = [xs, ys, w, h];
        ax = axes('Position', pos);
        h = imagesc(We{end});
          set(h,'AlphaData',0.5); 
          axis square;
          ax = gca;
          ax.XAxisLocation = 'top';
          yticks(1:6);
          xticks(1:6);
          xticklabels({});
          yticklabels({});
          colormap(ax,cmap);
          caxis([0 1]);

        xs = 0.39;
        ys = 0.74;
        w = 0.20;
        sp = 0.025;
        for j=1:2
          for i=1:3
            pos = [xs + (w*0.85)*(i-1), ys - (w + sp)*(j-1) , w, w];
            ax = axes('Position',pos);
            h = imagesc(rank1_mats{end}{i + 3*(j-1)});
            set(h,'AlphaData',0.5);
              axis square;
              ax = gca;
              ax.XAxisLocation = 'top';
              yticklabels([])
              xticklabels([])
              colormap(ax,cmap);
              caxis([-1 1]);
          end
        end


        pos = [xs, 0.4, 3*w-2*sp, w];
        ann = annotation('textbox','Position',pos);
        ann.String = 'Rank 1 matrices';
        ann.EdgeColor = 'none';
        ann.HorizontalAlignment = 'center';
        ann.VerticalAlignment = 'bottom';

      xs = 0.92;
      ys = 0.51;
      w = 0.025;
      h = 0.43;
      sp = 0.025;
      pos = [xs, ys, w, h];
      cb_ax = axes('Position',pos);
      v = linspace(-1,1,n)';             % vertical color strip
      h = imagesc([-1 1],[-1 1],v);         % draw gradient
      colormap(cb_ax,cmap);
      set(h,'AlphaData',0.5);                % transparency
      cb_ax.YDir = 'normal';
      cb_ax.XAxis.Visible = 'off';
      cb_ax.XTick = [];
      cb_ax.YTick = [-1 0 1];
      cb_ax.YTickLabel = {'-1','0','1'};
      cb_ax.YAxisLocation = 'right';  % moves y ticks to the right
      box(cb_ax,'off');






        xs = 0.42;
        ys = 0.1;
        w = 0.48;
        h = 0.3;
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
          p.MarkerSize = 1;
          p.LineWidth = 1;
          p.Color = trace_list{i}{2};
        end
        axis tight;
        ylim([0,1]);
        yticklabels({});
        xticklabels({});
        xlabel('sim time');
        %ylabel({'Explained', 'Variance'})
        ax.Color = 'none';


        mode_colors = cell2mat( ...
          cellfun(@(c) c{2}, trace_list, 'UniformOutput', false)');


        xs = 0.07;
        ys = 0.1;
        w = 0.33;
        h = 0.30;
        pos = [xs, ys, w, h];
        ax = axes('Position',pos);
          b = bar(explained_var(end,:), 'FaceColor','flat', 'LineWidth', 1);
          b.CData = mode_colors;
          b.EdgeColor = 'none';
          hold all;
          p = plot(sum_of_var(end,:), '--ok', 'LineWidth', 1);
          p.MarkerSize = 4;
          p.MarkerFaceColor = 'k';
          ylim([0,1]);
          yticklabels({'0','','1'});
          ylab = ylabel('EV');
          ylab.Position  = [-0.6   0.5000   -1.0000];
        ax.Color = 'none';


%        % SVD Analysis
%        xs = 0.15;
%        ys = 0.3;
%        w = 0.45;
%        h = 0.45;
%        pos = [xs, ys, w, h];
%        ax = axes('Position', pos);
%          h = imagesc(We{1});
%          set(h,'AlphaData',0.5); 
%          axis square;
%          ax = gca;
%          ax.XAxisLocation = 'top';
%          yticks(1:6);
%          xticks(1:6)
%          yticklabels({'MN-SF', 'MN-SE', 'MN-EF', 'MN-EE', 'MN-BF', 'MN-BE'});
%          xticklabels({'Ia-SF', 'Ia-SE', 'Ia-EF', 'Ia-EE', 'Ia-BF', 'Ia-BE'});
%          xtickangle(45);
%          colormap(ax,cmap);
%          caxis([0 1]);
%          ax.XAxis.FontSize = 10;
%          ax.YAxis.FontSize = 10;
%          label_pos = pos .* [1 0.925 1 0.1];
%          ann = annotation('textbox','Position',label_pos);
%          ann.String = 'Initial';
%          ann.EdgeColor = 'none';
%          ann.HorizontalAlignment = 'center';
%
%      xshift = 0.35;
%      pos(1) = pos(1) + xshift;
%      ax = axes('Position',pos);
%        h = imagesc(We{end});
%        set(h,'AlphaData',0.5); 
%        axis square;
%        ax = gca;
%        ax.XAxisLocation = 'top';
%        yticklabels([])
%        xticklabels([])
%        colormap(ax,cmap);
%        caxis([0 1]);
%        %colorbar;
%
%        %label_pos = pos .* [1 0.925 1 0.1];
%        %ann = annotation('textbox','Position',label_pos);
%        %ann.String = 'Final';
%        %ann.EdgeColor = 'none';
%        %ann.HorizontalAlignment = 'center';
%
%      xshift = 0.4;
%      pos(1) = pos(1) + xshift;
%      pos(3) = 0.025;
%      %cb = colorbar('Position',pos);
%      %colormap(cmap);
%      %caxis([0 1]);
%
%      cb_ax = axes('Position',pos);
%      v = linspace(0,1,n)';             % vertical color strip
%      h = imagesc([0 1],[0 1],v);         % draw gradient
%      colormap(cb_ax,cmap);
%      set(h,'AlphaData',0.5);                % transparency
%      cb_ax.YDir = 'normal';
%      cb_ax.XTick = [];
%      cb_ax.YTick = [0 0.5 1];
%      cb_ax.YTickLabel = {'0','0.5','1'};
%      cb_ax.YAxisLocation = 'right';  % moves y ticks to the right
%      box(cb_ax,'off');
%
%        % W decompose legend
%        xs = 0;
%        ys = 0.03;
%        w = 0.2;
%        h = 0.2;
%        shift1 = 0.21;
%        shift2 = 0.0;
%        shift3 = -0.37;
%        ax = axes('Position',[xs, ys, w, w]);
%%        p = imagesc( ...
%%          mRf ...
%%          + 0.70*mRe ...
%%          + 0.30*mSf ...
%%          + 0.10*mSe ...
%%          - 0.75*mAf ...
%%          - mAe);
%        p = imagesc( ...
%          + 0.7*mRf ...
%          + 0.7*mRe ...
%          + 0.1*mSf ...
%          + 0.1*mSe ...
%          - 1*mAf ...
%          - 1*mAe);
%        p.AlphaData = 0.8;
%        axis square;
%        axis off;
%        colormap(ax,'default');
%        caxis([-1 1]);
%
%%        legend_list = { {1, 'Flexion Reflex'}, ...
%%                        {0.7, 'Extension Reflex'}, ...
%%                        {0.3, 'Flexion Synergy'}, ...
%%                        {0.1, 'Extension Synergy'}, ...
%%                        {-0.75, 'Flexion Antagonist'}, ...
%%                        {-1, 'Extension Antagonist'} };
%
%        legend_list = { {0.7, 'Reflex'}, ...
%                        {0.1, 'Synergy'}, ...
%                        {-1, 'Anti'} };
%
%        ys = 0.56;
%        space = 0.055;
%
%        rgb_val = zeros(numel(legend_list),3);
%        for i = 1:length(legend_list)
%
%
%          ax = axes('Position',[xs + shift1-0.03, ys+shift3-0.03, 0.045, 0.045]);
%          p = imagesc(legend_list{i}{1});
%          axis square;
%          yticklabels([])
%          xticklabels([])
%          caxis([-1 1]);
%
%          c = legend_list{i}{1};
%          rgb_map = colormap(ax,'default');
%          c_range = caxis(ax);
%          idx = round((c - c_range(1)) / diff(c_range) * (size(rgb_map,1) - 1)) + 1;
%          idx = max(1, min(idx, size(rgb_map,1)));  % clamp to valid range
%          rgb_val(i,:) = rgb_map(idx, :);
%
%          ann = annotation('textbox','Position',[xs+shift1+shift2 ys+shift3 0.5 0.045]);
%          ann.String = legend_list{i}{2};
%          ann.EdgeColor = 'none';
%          ann.HorizontalAlignment = 'left';
%          ys = ys - space;
%        end
%
%        % Time series traces
%        xs = 0.16 + xshift;
%        ys = 0.09;
%        w = 0.33;
%        h = 0.17;
%        pos = [xs, ys, w, h];
%        ax = axes('Position',pos);
%%        trace_list = { ...
%%          {Rf_a, rgb_val(1,:)}, ...
%%          {Re_a, rgb_val(2,:)}, ...
%%          {Sf_a, rgb_val(3,:)}, ...
%%          {Se_a, rgb_val(4,:)}, ...
%%          {Af_a, rgb_val(5,:)}, ...
%%          {Ae_a, rgb_val(6,:)}, ...
%%        };
%%        trace_list = { ...
%%          {R_a, rgb_val(2,:)}, ...
%%          {S_a, rgb_val(4,:)}, ...
%%          {A_a, rgb_val(6,:)}, ...
%%        };
%        trace_list = { ...
%          {R_a, rgb_val(1,:)}, ...
%          {S_a, rgb_val(2,:)}, ...
%          {A_a, rgb_val(3,:)}, ...
%        };
%        for i=1:numel(trace_list)
%          p = plot(trace_list{i}{1},'-o'); hold all;
%          p.MarkerSize = 0.1;
%          p.LineWidth = 1
%          p.Color = trace_list{i}{2};
%        end
%        axis tight;
%        ylim([0,1]);
%        xl = xlim;
%        xticklabels({});
%        ax.YTick = [0 0.5 1];
%        ax.YTickLabel = {'0','', '1'};
%        ylabel('Motif')
%        xlabel('sim time');
%        ax.Color = 'none';
%
%%return
%%        % W decompose legend
%%        xs = 0.31;
%%        w = 0.32;
%%        ys = 0.63;
%%        shift1 = 0.09;
%%        shift2 = 0.04;
%%        ax = axes('Position',[xs, ys, w, w]);
%%%        p = imagesc( ...
%%%          mRf ...
%%%          + 0.70*mRe ...
%%%          + 0.30*mSf ...
%%%          + 0.10*mSe ...
%%%          - 0.75*mAf ...
%%%          - mAe);
%%        p = imagesc( ...
%%          + 0.7*mRf ...
%%          + 0.7*mRe ...
%%          + 0.1*mSf ...
%%          + 0.1*mSe ...
%%          - 1*mAf ...
%%          - 1*mAe);
%%        p.AlphaData = 0.8;
%%        axis square;
%%        axis off;
%%        caxis([-1 1]);
%%
%%%        legend_list = { {1, 'Flexion Reflex'}, ...
%%%                        {0.7, 'Extension Reflex'}, ...
%%%                        {0.3, 'Flexion Synergy'}, ...
%%%                        {0.1, 'Extension Synergy'}, ...
%%                        {-0.75, 'Flexion Antagonist'}, ...
%%                        {-1, 'Extension Antagonist'} };
%
%        legend_list = { {0.7, 'Reflex'}, ...
%                        {0.1, 'Synergy'}, ...
%                        {-1, 'Anti'} };
%
%        ys = 0.56;
%        space = 0.055;
%
%        rgb_val = zeros(numel(legend_list),3);
%        for i = 1:length(legend_list)
%          ax = axes('Position',[xs + shift1, ys, 0.045, 0.045]);
%          p = imagesc(legend_list{i}{1});
%          axis square;
%          yticklabels([])
%          xticklabels([])
%          caxis([-1 1]);
%
%          c = legend_list{i}{1};
%          rgb_map = colormap(ax);
%          c_range = caxis(ax);
%          idx = round((c - c_range(1)) / diff(c_range) * (size(rgb_map,1) - 1)) + 1;
%          idx = max(1, min(idx, size(rgb_map,1)));  % clamp to valid range
%          rgb_val(i,:) = rgb_map(idx, :);
%
%          ann = annotation('textbox','Position',[xs+shift1+shift2 ys 0.5 0.045]);
%          ann.String = legend_list{i}{2};
%          ann.EdgeColor = 'none';
%          ann.HorizontalAlignment = 'left';
%          ys = ys - space;
%        end
%
%        % Time series traces
%        xs = 0.05;
%        ys = 0.60;
%        w = 0.35;
%        h = 0.22;
%        pos = [xs, ys, w, h];
%        ax = axes('Position',pos);
%%        trace_list = { ...
%%          {Rf_a, rgb_val(1,:)}, ...
%%          {Re_a, rgb_val(2,:)}, ...
%%          {Sf_a, rgb_val(3,:)}, ...
%%          {Se_a, rgb_val(4,:)}, ...
%%          {Af_a, rgb_val(5,:)}, ...
%%          {Ae_a, rgb_val(6,:)}, ...
%%        };
%%        trace_list = { ...
%%          {R_a, rgb_val(2,:)}, ...
%%          {S_a, rgb_val(4,:)}, ...
%%          {A_a, rgb_val(6,:)}, ...
%%        };
%        trace_list = { ...
%          {R_a, rgb_val(1,:)}, ...
%          {S_a, rgb_val(2,:)}, ...
%          {A_a, rgb_val(3,:)}, ...
%        };
%        for i=1:numel(trace_list)
%          p = plot(trace_list{i}{1},'-o'); hold all;
%          p.MarkerSize = 2;
%          p.Color = trace_list{i}{2};
%        end
%        xticklabels({});
%        axis tight;
%        ylim([0,1]);
%        ylabel({'Motif Strength','||W * M|| / ||M||'})
%        ax.Color = 'none';
%
%        ys = 0.35;
%        pos = [xs, ys, w, h];
%        ax = axes('Position',pos);
%%        trace_list = { ...
%%          {gradient(R_a), rgb_val(2,:)}, ...
%%          {gradient(S_a), rgb_val(4,:)}, ...
%%          {gradient(A_a) rgb_val(6,:)}, ...
%%        };
%        trace_list = { ...
%          {gradient(R_a), rgb_val(1,:)}, ...
%          {gradient(S_a), rgb_val(2,:)}, ...
%          {gradient(A_a) rgb_val(3,:)}, ...
%        };
%        for i=1:numel(trace_list)
%          p = plot(trace_list{i}{1},'-o'); hold all;
%          p.MarkerSize = 2;
%          p.Color = trace_list{i}{2};
%        end
%        axis tight;
%        xticklabels({});
%        ylim([0,1]);
%        ylabel('dAbsolute')
%        ax.Color = 'none';
%
%
%
%        ys = 0.1;
%        pos = [xs, ys, w, h];
%        ax = axes('Position',pos);
%        trace_list = { ...
%          {explained_var(:,1), [1 0.1 0.1]}, ...
%          {explained_var(:,2), [0.8 0.1 0.1]}, ...
%          {explained_var(:,3), [0.6 0.1 0.1]}, ...
%          {explained_var(:,4), [0.4 0.1 0.1]}, ...
%          {explained_var(:,5), [0.2 0.1 0.1]}, ...
%          {explained_var(:,6), [0 0.1 0.1]}, ...
%        };
%        for i=1:numel(trace_list)
%          p = plot(trace_list{i}{1},'-o'); hold all;
%          p.MarkerSize = 2;
%          p.Color = trace_list{i}{2};
%        end
%        axis tight;
%        ylim([0,1]);
%        xlabel('Simulation Time (s)');
%        ylabel('Explained Variance')
%        ax.Color = 'none';
%
%        ys = 0.85;
%        h = 0.1;
%        pos = [xs, ys, w, h];
%        ax = axes('Position',pos);
%        plot(muscle_strength,'-k');
%        xticklabels({});
%        axis tight;
%        ylim([0,1]);
%        ylabel('Strength')
%        ax.Color = 'none';
%
%
%        % SVD Analysis
%        xs = 0.45;
%        ys = 0.47;
%        w = 0.45;
%        h = 0.45;
%        pos = [xs, ys, w, h];
%        ax = axes('Position', pos);
%          imagesc(We{1});
%          axis square;
%          ax = gca;
%          ax.XAxisLocation = 'top';
%          yticks(1:6);
%          yticklabels({'MN-SF', 'MN-SE', 'MN-EF', 'MN-EE', 'MN-BF', 'MN-BE'});
%          xticklabels({'Ia-SF', 'Ia-SE', 'Ia-EF', 'Ia-EE', 'Ia-BF', 'Ia-BE'});
%          colormap(ax,cmap);
%          caxis([0 1]);
%          ax.XAxis.FontSize = 12;
%          ax.YAxis.FontSize = 12;
%
%
%
%          pause
%          label_pos = pos .* [1 0.925 1 0.1];
%          ann = annotation('textbox','Position',label_pos);
%          ann.String = 'Initial';
%          ann.EdgeColor = 'none';
%          ann.HorizontalAlignment = 'center';
%
%      xshift = 0.2;
%      pos(1) = pos(1) + xshift;
%      ax = axes('Position',pos);
%        imagesc(We{end});
%        axis square;
%        ax = gca;
%        ax.XAxisLocation = 'top';
%        yticklabels([])
%        xticklabels([])
%        colormap(ax,cmap);
%        caxis([0 1]);
%        colorbar;
%
%        label_pos = pos .* [1 0.925 1 0.1];
%        ann = annotation('textbox','Position',label_pos);
%        ann.String = 'Final';
%        ann.EdgeColor = 'none';
%        ann.HorizontalAlignment = 'center';
%
%        xs = 0.52;
%        w = 0.15;
%        sp = 0.015;
%        for j=1:2
%          for i=1:3
%            pos = [xs + (w*0.45)*(i-1), 0.25 - (w + sp)*(j-1) , w, w];
%            ax = axes('Position',pos);
%            imagesc(rank1_mats{end}{i + 3*(j-1)});
%              axis square;
%              ax = gca;
%              ax.XAxisLocation = 'top';
%              yticklabels([])
%              xticklabels([])
%              colormap(ax,cmap);
%              caxis([-1 1]);
%          end
%        end
%
%        pos = [xs + 0.05, 0.025, w, w];
%        ann = annotation('textbox','Position',pos);
%        ann.String = 'Rank 1 matrices';
%        ann.EdgeColor = 'none';
%        ann.HorizontalAlignment = 'center';
%        ann.VerticalAlignment = 'bottom';
%
%        xs = 0.8;
%        ys = 0.05;
%        w = 0.15;
%        h = 0.35;
%        pos = [xs, ys, w, h];
%        ax = axes('Position',pos);
%          bar(explained_var(end,:), 'k', 'LineWidth', 2);
%          hold all;
%          plot(sum_of_var(end,:), '-or', 'LineWidth', 2);
%          ylabel('Explained Variance');
%          ylim([0,1]);
%        ax.Color = 'none';
%
%        xs = 0.41;
%        ann = annotation('textbox','Position',[xs,0,0.5,0.25]);
%        ann.Interpreter = 'none';
%        ann.String = keys_str;
%        ann.EdgeColor = 'none';
%        ann.HorizontalAlignment = 'left';
%        ann.VerticalAlignment = 'top';
%        ann.FontSize = 7;
%
%        ann = annotation('textbox','Position',[xs,0,0.12,0.25]);
%        ann.Interpreter = 'none';
%        ann.String = vals_str;
%        ann.EdgeColor = 'none';
%        ann.HorizontalAlignment = 'right';
%        ann.VerticalAlignment = 'top';
%        ann.FontSize = 7;
%
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
