function summary_table = ds_process_babble(root,opt)
  arguments
    root
    opt.FigVis = 'on';
    opt.printfig (1,1) logical = false
    opt.outdir (1,1) string = 'analysis'
    opt.last_n_samples = 100;
    opt.testrun = false
  end

  tic
  set(0, 'DefaultFigureVisible', opt.FigVis);

  savepath = [root,'/',char(opt.outdir)];
  dashboard_path = [savepath,'/dashboard'];
  summary_path = [savepath,'/summary'];
  dirs_created = false;

  dirs = dir([root,'/*']);

  measures = {@mean, @var};
  measures_str = {'mean', 'var'};
  signals_str = { ...
    'rS', 'rE', 'rB', 'r', ...
    'sS', 'sE', 'sB', 's', ...
    'aS', 'aE', 'aB', 'a', ...
    'score'
  };

  ncols_outcomes = numel(signals_str) * numel(measures);
  rows = {};
  label = {};
  count = 0;

  for i = 1:length(dirs)
    name = dirs(i).name;
    if dirs(i).isdir && ~any(strcmp(name, {'.', '..', '.DS_Store'}))
      path = [root,'/',name];
      if ~exist([path,'/data.txt'], 'file')
          fprintf(['\nSkipping: ', path,'\n']);
          continue;
      end
      fprintf(['\nProcessing: ', path,'\n']);

      % Create output directories on first valid sim
      if ~dirs_created
        if ~exist(savepath, 'dir'), mkdir(savepath); end
        if opt.printfig
          if ~exist(dashboard_path, 'dir'), mkdir(dashboard_path); end
        end
        if ~exist(summary_path, 'dir'), mkdir(summary_path); end
        dirs_created = true;
      end

      % Extract parameters
      [keys, vals] = extract_parameters(path);

      % Extract data
      [episode, time, muscle_strength, Wi, We] = ...
          extract_data([path,'/data.txt']);
      %save('We.mat','We');
      motif = ds_Ia_motif(We);

      % Outcome Measures
      signals = { ...
        motif.rp(:,1), motif.rp(:,2), motif.rp(:,3), motif.rg, ...
        motif.sp(:,1), motif.sp(:,2), motif.sp(:,3), motif.sg, ...
        motif.ap(:,1), motif.ap(:,2), motif.ap(:,3), motif.ag ...
        motif.score
      };

      outcomes = ...
        cellfun( @(signal) ...
          cellfun(@(measure) ...
                  measure(signal((end-opt.last_n_samples):end)), ...
                  measures, 'UniformOutput', false), ...
          signals, 'UniformOutput', false);

      if ~isempty(keys)
        param_cells = num2cell(str2double(vals(:)'));
      else
        param_cells = {};
      end

      count = count + 1;
      rows{count} = [{name}, param_cells, outcomes{:}];

      % Build column labels once
      if count == 1
        outcomes_str = cellfun( ...
          @(signal_str) cellfun( ...
            @(measure) [signal_str, '_', measure], ...
            measures_str, 'UniformOutput', false), ...
          signals_str, 'UniformOutput',false);
        label = [{'sim-ID'}, keys(:)', outcomes_str{:}];
      end

      % Dashboard figure
      if opt.printfig
        ds_plot_dashboard(...
          We,muscle_strength,motif, ...
          strjoin(keys,newline),strjoin(vals,newline),name);

        if opt.testrun
          summary_table =[];
          return
        end

        if ~isempty(keys)
          para_str = cellfun(@(v) ['_',v],vals,'UniformOutput',false);
        else
          para_str = {};
        end
        sim_fig_path = [ ...
          dashboard_path, ...
          '/', name, ...
          para_str{:}, ...
          '.png'];
        fprintf(['Saving...\n']);
        print(gcf,'-dpng',sim_fig_path);
        fprintf(['Saved figure to: ', sim_fig_path, '\n\n']);
        close(gcf);
      end
    end
  end

  % Build table from collected rows
  if count > 0
    numel(vertcat(rows{:}))
    numel(label)
    summary_table = cell2table(vertcat(rows{:}), 'VariableNames', label);
    save([summary_path,'/summary_table.mat'],'summary_table');
  else
    summary_table = table();
    fprintf('\nNo valid simulations found.\n');
  end
  set(0, 'DefaultFigureVisible', 'on');
  toc
end
