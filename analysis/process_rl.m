function results_table = process_rl(log_root, opt)
  arguments
    log_root (1,1) string
    opt.show = 'off'
    opt.abscissa = 'step'
    opt.mean_win = 500
    opt.ic_window_mode = 'all'
    opt.ic_win = 500
    opt.rew_var (1,1) string = 'ep_len'
    opt.success_thresh = []  % unused; kept for back-compat
    opt.top_n_models = 0
    opt.ic_type (1,1) string = 'joint'
    opt.do_kernel = true
    opt.table_fields = {}
    opt.use_spine_labs = false
    opt.outdir (1,1) string = 'analysis'
    opt.fig1_only (1,1) logical = false
    opt.eval (1,1) logical = true
    opt.check_ic (1,1) logical = false
    opt.html_report (1,1) logical = true
  end
  tic
  savepath = fullfile(log_root, opt.outdir);
  if ~exist(savepath, 'dir'), mkdir(savepath); end
  diary(fullfile(savepath, 'runtime.log'));

  set(0, 'DefaultFigureVisible', opt.show);

  %% Constants
  c1map = cool;
  markersize_boxplot = 40;
  markersize = 20;
  percent_threshold = 0.1;
  peak_smooth_win = 50;
  isoview = [0, 90];
  L = 1;

  %% Load data
  df_path = fullfile(log_root, 'df.h5');
  Step = h5read(df_path, '/table/Step');
  Value = h5read(df_path, '/table/Value');
  Tag = cellstr(h5read(df_path, '/table/Tag'));
  ModelID = cellstr(h5read(df_path, '/table/ModelID'));
  Model = cellstr(h5read(df_path, '/table/Model'));

  T = table(Step, Value, Tag, ModelID, Model);

  all_models = unique(T.ModelID);
  n_models = numel(all_models);

  %% Group by config — read custom.txt for each sim
  config_keys = cell(n_models, 1);
  skip_keys = {'log_dir', 'options_seed', 'options_check_points', ...
               'options_total_sim_steps'};
  for i = 1:n_models
    parts = split(all_models{i}, '-');
    sim_id = parts{1};
    custom_path = fullfile(log_root, sim_id, 'custom.txt');
    fid = fopen(custom_path, 'r');
    if fid ~= -1
      raw = fread(fid, inf, 'uint8=>char')'; fclose(fid);
      p = jsondecode(raw);
      leaves = extract_leaves(p);
      spine_type = parts{2};
      kv_parts = {sprintf('spine=%s', spine_type)};
      for t = 1:size(leaves, 1)
        k = leaves{t, 1};
        if any(strcmp(k, skip_keys)); continue; end
        v = leaves{t, 2};
        if isnumeric(v) && numel(v) > 1
          kv_parts{end+1} = sprintf('%s=[%s]', k, ...
            strjoin(arrayfun(@(x) sprintf('%g',x), v, 'Uni', false), ','));
        elseif isnumeric(v)
          kv_parts{end+1} = sprintf('%s=%g', k, v);
        else
          kv_parts{end+1} = sprintf('%s=%s', k, string(v));
        end
      end
      config_keys{i} = strjoin(kv_parts, ', ');
    else
      % Fallback: parse TB directory name
      config_keys{i} = strjoin(parts(2:end), '-');
    end
  end

  [unique_configs, ~, group_idx] = unique(config_keys);
  n_groups = numel(unique_configs);

  % Build short display labels from only the fields that differ across groups
  if n_groups > 1
    % Parse each unique config back into key=value tokens
    tok = cellfun(@(s) strsplit(s, ', '), unique_configs, 'Uni', false);
    % Find tokens that are NOT identical across all groups
    common = tok{1};
    for g = 2:n_groups
      common = intersect(common, tok{g}, 'stable');
    end
    config_labels = cell(n_groups, 1);
    for g = 1:n_groups
      diff_tok = setdiff(tok{g}, common, 'stable');
      % Shorten keys: strip leading hierarchy (e.g. env_kwargs_reward_type -> reward_type)
      short = cell(size(diff_tok));
      for d = 1:numel(diff_tok)
        eq_pos = find(diff_tok{d} == '=', 1);
        if isempty(eq_pos)
          short{d} = diff_tok{d};
          continue;
        end
        k_str = diff_tok{d}(1:eq_pos-1);
        v_str = diff_tok{d}(eq_pos+1:end);
        parts_k = strsplit(k_str, '_');
        if numel(parts_k) >= 2
          short{d} = [strjoin(parts_k(end-1:end), '_') '=' v_str];
        else
          short{d} = [k_str '=' v_str];
        end
      end
      config_labels{g} = strjoin(short, ', ');
    end
  else
    config_labels = unique_configs;
  end

  cmap = lines(n_groups);
  if (n_groups == 2) && opt.use_spine_labs
    cmap = [ [1 0 0]; [0 0 0] ];
  end

  %% ============================================================
  %% COMPUTE
  %% ============================================================

  %% Per-model metrics for ranking
  model_n_perfect = zeros(n_models, 1);
  model_mean_rew = zeros(n_models, 1);
  model_max_mean_rew = zeros(n_models, 1);

  for i = 1:n_models
    [~, rew] = get_model_tag(T, all_models{i}, sprintf('eps/%s', opt.rew_var));
    if isempty(rew)
      [~, rew] = get_model_tag(T, all_models{i}, 'eps/ep_rew');
    end
    [~, solved] = get_model_tag(T, all_models{i}, 'eps/solved');
    model_n_perfect(i) = max(solved);
    model_mean_rew(i) = mean(rew);
    model_max_mean_rew(i) = max(movmean(rew, opt.mean_win));
  end

  %% Model selection (top N per group)
  model_selected = false(n_models, 1);
  for g = 1:n_groups
    g_mask = group_idx == g;
    g_indices = find(g_mask);
    g_n_perfect = model_n_perfect(g_mask);

    if opt.top_n_models > 0 && opt.top_n_models < sum(g_mask)
      [~, sorted_idx] = sort(g_n_perfect, 'descend');
      selected_idx = g_indices(sorted_idx(1:opt.top_n_models));
    else
      selected_idx = g_indices;
    end
    model_selected(selected_idx) = true;
  end

  %% Console: group summary
  fprintf('Found %d models in %d groups:\n', n_models, n_groups);
  if opt.top_n_models > 0
    fprintf('Using top %d models per group (by # Perfect):\n', opt.top_n_models);
  end
  for g = 1:n_groups
    g_mask = group_idx == g;
    g_selected = model_selected & g_mask;
    fprintf('  [%d] %s (n=%d, selected=%d)\n', g, config_labels{g}, ...
      sum(g_mask), sum(g_selected));
    g_indices = find(g_mask);
    for j = 1:length(g_indices)
      idx = g_indices(j);
      sel_str = '';
      if model_selected(idx), sel_str = ' <-- SELECTED'; end
      fprintf('      Model %d: #Perfect=%d, MaxMean=%.1f%s\n', ...
        idx, model_n_perfect(idx), model_max_mean_rew(idx), sel_str);
    end
  end

  %% Traces cache — extract all TB data once for selected models
  sel_indices = find(model_selected);
  n_sel = numel(sel_indices);

  traces = struct('step',{},'solved',{},'perfect',{},'eps',{},...
                  'rew_step',{},'rew',{},'roll_mean',{}, ...
                  'ic_q0',{},'ic_q1',{},'ic_pend',{},'fc_q0',{},'fc_q1',{}, ...
                  'eval_step',{},'eval_n_perfect',{},'eval_mean_rew',{}, ...
                  'decimation_factor',{},'model_idx',{});
  for k = 1:n_sel
    i = sel_indices(k);
    [traces(k).step, traces(k).solved] = ...
      get_model_tag(T, all_models{i}, 'eps/solved');
    traces(k).perfect = logical([0; diff(double(traces(k).solved)) > 0]);
    [~, traces(k).eps] = get_model_tag(T, all_models{i}, 'eps/eps');
    [traces(k).rew_step, traces(k).rew] = ...
      get_model_tag(T, all_models{i}, sprintf('eps/%s', opt.rew_var));
    if isempty(traces(k).rew)
      [traces(k).rew_step, traces(k).rew] = ...
        get_model_tag(T, all_models{i}, 'eps/ep_rew');
    end
    traces(k).roll_mean = movmean(traces(k).rew, opt.mean_win);
    [~, traces(k).ic_q0]   = get_model_tag(T, all_models{i}, 'eps/ic_q0');
    [~, traces(k).ic_q1]   = get_model_tag(T, all_models{i}, 'eps/ic_q1');
    [~, traces(k).ic_pend] = get_model_tag(T, all_models{i}, 'eps/ic_pend');
    [~, traces(k).fc_q0] = get_model_tag(T, all_models{i}, 'eps/fc_q0');
    [~, traces(k).fc_q1] = get_model_tag(T, all_models{i}, 'eps/fc_q1');
    [traces(k).eval_step, traces(k).eval_n_perfect] = ...
      get_model_tag(T, all_models{i}, 'eval/n_perfect');
    [~, traces(k).eval_mean_rew] = ...
      get_model_tag(T, all_models{i}, 'eval/mean_rew');
    traces(k).model_idx = i;
    % Load decimation_factor from config.json (full resolved config)
    parts_k = split(all_models{i}, '-');
    cfg_path = fullfile(log_root, parts_k{1}, 'config.json');
    fid_cfg = fopen(cfg_path, 'r');
    if fid_cfg ~= -1
      raw_cfg = fread(fid_cfg, inf, 'uint8=>char')'; fclose(fid_cfg);
      cfg_data = jsondecode(raw_cfg);
      traces(k).decimation_factor = cfg_data.supraspine_kwargs.decimation_factor;
    else
      traces(k).decimation_factor = 1;
    end
  end

  %% IC sequence check — verify same seed produces same IC sequence across configs
  if opt.check_ic
    % Read seed for each model from config.json
    model_seeds = nan(n_sel, 1);
    for k = 1:n_sel
      i = sel_indices(k);
      parts_k = split(all_models{i}, '-');
      cfg_path = fullfile(log_root, parts_k{1}, 'config.json');
      fid = fopen(cfg_path, 'r');
      if fid ~= -1
        raw = fread(fid, inf, 'uint8=>char')'; fclose(fid);
        cfg = jsondecode(raw);
        model_seeds(k) = cfg.options.seed;
      end
    end

    unique_seeds = unique(model_seeds(~isnan(model_seeds)));
    ns = numel(unique_seeds);
    colors = lines(n_sel);

    fig_ic = figure('Name','IC Sequence Check', ...
      'Position',[100 400 800 250*ns]);

    for s = 1:ns
      seed_mask = model_seeds == unique_seeds(s);
      seed_idx  = find(seed_mask);
      if numel(seed_idx) < 2, continue; end

      % q0 row
      ax0 = subplot(ns, 2, (s-1)*2 + 1);
      hold(ax0, 'on');
      % q1 row
      ax1 = subplot(ns, 2, (s-1)*2 + 2);
      hold(ax1, 'on');

      for j = 1:numel(seed_idx)
        k   = seed_idx(j);
        n   = numel(traces(k).ic_q0);
        eps = 1:n;
        plot(ax0, eps, traces(k).ic_q0, 'Color', [colors(k,:) 0.7], 'LineWidth', 0.8);
        plot(ax1, eps, traces(k).ic_q1, 'Color', [colors(k,:) 0.7], 'LineWidth', 0.8);
      end

      ylabel(ax0, 'ic\_q0'); ylabel(ax1, 'ic\_q1');
      title(ax0, sprintf('seed %d', unique_seeds(s)));
      title(ax1, sprintf('seed %d', unique_seeds(s)));

      % Console: max abs diff between first model and all others
      ref = seed_idx(1);
      for j = 2:numel(seed_idx)
        cmp = seed_idx(j);
        n = min(numel(traces(ref).ic_q0), numel(traces(cmp).ic_q0));
        d0 = max(abs(traces(ref).ic_q0(1:n) - traces(cmp).ic_q0(1:n)));
        d1 = max(abs(traces(ref).ic_q1(1:n) - traces(cmp).ic_q1(1:n)));
        fprintf('seed=%d  max_diff_q0=%.2e  max_diff_q1=%.2e\n', ...
          unique_seeds(s), d0, d1);
      end
    end

    xlabel(ax0, 'episode'); xlabel(ax1, 'episode');
    sgtitle(fig_ic, 'IC Sequence Check — overlapping lines = matched seeding');
  end

  %% Per-group metrics accumulation
  metrics = struct();
  for g = 1:n_groups
    metrics(g).solved_max = [];
    metrics(g).reward_max = [];
    metrics(g).mean_rew = [];
    metrics(g).max_mean_rew = [];
    metrics(g).eps_to_perfect = [];
    metrics(g).supra_steps_to_perfect = [];
    metrics(g).sim_steps_to_perfect = [];
    metrics(g).hull_area = 0;
    metrics(g).eval_n_perfect_max = [];
    metrics(g).eval_mean_rew_max  = [];
  end

  for k = 1:n_sel
    i = traces(k).model_idx;
    g = group_idx(i);
    metrics(g).solved_max(end+1) = max(traces(k).solved);
    metrics(g).mean_rew(end+1) = model_mean_rew(i);
    metrics(g).max_mean_rew(end+1) = model_max_mean_rew(i);
    metrics(g).reward_max(end+1) = max(traces(k).roll_mean);
    idx_first = find(traces(k).solved > 0, 1, 'first');
    if ~isempty(idx_first)
      metrics(g).eps_to_perfect(end+1)        = traces(k).eps(idx_first);
      metrics(g).supra_steps_to_perfect(end+1) = traces(k).step(idx_first);
      metrics(g).sim_steps_to_perfect(end+1)   = traces(k).step(idx_first) * traces(k).decimation_factor;
    else
      metrics(g).eps_to_perfect(end+1)        = NaN;
      metrics(g).supra_steps_to_perfect(end+1) = NaN;
      metrics(g).sim_steps_to_perfect(end+1)   = NaN;
    end
    if ~isempty(traces(k).eval_n_perfect)
      metrics(g).eval_n_perfect_max(end+1) = max(traces(k).eval_n_perfect);
      metrics(g).eval_mean_rew_max(end+1)  = max(traces(k).eval_mean_rew);
    end
  end

  %% IC windowed data
  all_q0 = []; all_q1 = []; all_pend = []; all_rew = []; all_group = []; all_perfect = false(0,1);

  for k = 1:n_sel
    g = group_idx(traces(k).model_idx);
    rew = traces(k).rew;
    q0 = traces(k).ic_q0;
    q1 = traces(k).ic_q1;
    n = numel(rew);

    if strcmp(opt.ic_window_mode, 'all')
      start_idx = 1;
      end_idx = n;
    elseif strcmp(opt.ic_window_mode, 'peak')
      roll_rew = movmean(rew, peak_smooth_win);
      [~, peak_idx] = max(roll_rew);
      half_win = floor(opt.ic_win / 2);
      start_idx = peak_idx - half_win;
      end_idx = start_idx + opt.ic_win - 1;
      if start_idx < 1
        start_idx = 1;
        end_idx = opt.ic_win;
      end
      if end_idx > n
        end_idx = n;
        start_idx = n - opt.ic_win + 1;
      end
    else  % 'last'
      end_idx = n;
      start_idx = n - opt.ic_win + 1;
    end

    if ~strcmp(opt.ic_window_mode, 'all')
      if start_idx < 1 || (end_idx - start_idx + 1) ~= opt.ic_win
        error('Model %d: not enough episodes (%d) for window size %d', ...
          traces(k).model_idx, n, opt.ic_win);
      end
    end

    win_size = end_idx - start_idx + 1;
    all_q0      = [all_q0;      q0(start_idx:end_idx)];
    all_q1      = [all_q1;      q1(start_idx:end_idx)];
    all_rew     = [all_rew;     rew(start_idx:end_idx)];
    all_group   = [all_group;   g*ones(win_size,1)];
    all_perfect = [all_perfect; traces(k).perfect(start_idx:end_idx)];
    if ~isempty(traces(k).ic_pend)
      all_pend = [all_pend; traces(k).ic_pend(start_idx:end_idx)];
    else
      all_pend = [all_pend; zeros(win_size, 1)];
    end
  end

  q0_lim = [min(all_q0), max(all_q0)];
  q1_lim = [min(all_q1), max(all_q1)];
  rew_lim = [min(all_rew), max(all_rew)];

  %% Per-group hull area
  for g = 1:n_groups
    g_idx = all_group == g;
    g_q0 = all_q0(g_idx);
    g_q1 = all_q1(g_idx);
    g_perfect = all_perfect(g_idx);
    if sum(g_perfect) >= 3
      try
        [~, metrics(g).hull_area] = convhull(g_q0(g_perfect), g_q1(g_perfect));
      catch
        metrics(g).hull_area = NaN;
      end
    end
  end

  %% Band data (IC and FC)
  band_ic = struct('q0',cell(n_groups,1),'q1',cell(n_groups,1), ...
                   'rew',cell(n_groups,1));
  band_fc = struct('q0',cell(n_groups,1),'q1',cell(n_groups,1), ...
                   'rew',cell(n_groups,1));
  for g = 1:n_groups
    bic_q0 = []; bic_q1 = []; bic_rew = [];
    bfc_q0 = []; bfc_q1 = []; bfc_rew = [];
    for k = 1:n_sel
      if group_idx(traces(k).model_idx) ~= g, continue; end
      rew_norm = traces(k).rew / rew_lim(2);
      in_band = rew_norm >= percent_threshold & rew_norm < 1;
      bic_q0 = [bic_q0; traces(k).ic_q0(in_band)];
      bic_q1 = [bic_q1; traces(k).ic_q1(in_band)];
      bic_rew = [bic_rew; rew_norm(in_band)];
      bfc_q0 = [bfc_q0; traces(k).fc_q0(in_band)];
      bfc_q1 = [bfc_q1; traces(k).fc_q1(in_band)];
      bfc_rew = [bfc_rew; rew_norm(in_band)];
    end
    band_ic(g).q0 = bic_q0;
    band_ic(g).q1 = bic_q1;
    band_ic(g).rew = bic_rew;
    band_fc(g).q0 = bfc_q0;
    band_fc(g).q1 = bfc_q1;
    band_fc(g).rew = bfc_rew;
  end

  %% results_table struct
  results_table = struct();
  for g = 1:n_groups
    g_indices = find(group_idx == g);
    g_sim_ids = cell(numel(g_indices), 1);
    g_model_ids = cell(numel(g_indices), 1);
    for j = 1:numel(g_indices)
      g_model_ids{j} = all_models{g_indices(j)};
      parts = split(g_model_ids{j}, '-');
      g_sim_ids{j} = parts{1};
    end

    results_table(g).group = g;
    results_table(g).config = unique_configs{g};
    results_table(g).label = config_labels{g};
    results_table(g).sim_ids = g_sim_ids;
    results_table(g).model_ids = g_model_ids;
    results_table(g).n_perfect_mean = mean(metrics(g).solved_max);
    results_table(g).n_perfect_std = std(metrics(g).solved_max);
    results_table(g).max_mean_rew = mean(metrics(g).max_mean_rew);
    results_table(g).max_mean_rew_std = std(metrics(g).max_mean_rew);
    results_table(g).hull_area = metrics(g).hull_area;
    stp = metrics(g).eps_to_perfect;
    results_table(g).eps_to_perfect_mean = mean(stp, 'omitnan');
    results_table(g).eps_to_perfect_std  = std(stp, 'omitnan');
    stp = metrics(g).supra_steps_to_perfect;
    results_table(g).supra_steps_to_perfect_mean = mean(stp, 'omitnan');
    results_table(g).supra_steps_to_perfect_std  = std(stp, 'omitnan');
    stp = metrics(g).sim_steps_to_perfect;
    results_table(g).sim_steps_to_perfect_mean = mean(stp, 'omitnan');
    results_table(g).sim_steps_to_perfect_std  = std(stp, 'omitnan');

    % Parse params from custom.txt of first model in group
    first_model = all_models{g_indices(1)};
    sim_id = split(first_model, '-'); sim_id = sim_id{1};
    custom_path = fullfile(log_root, sim_id, 'custom.txt');
    fid = fopen(custom_path, 'r');
    if fid ~= -1
      raw = fread(fid, inf, 'uint8=>char')'; fclose(fid);
      p = jsondecode(raw);
      leaves = extract_leaves(p);
      skip = {'log_dir', 'options_seed'};
      for t = 1:size(leaves, 1)
        key = leaves{t, 1};
        if any(strcmp(key, skip)); continue; end
        field = strrep(key, '.', '_');
        field = regexprep(field, '[^a-zA-Z0-9_]', '');  % strip invalid field chars
        results_table(g).(field) = leaves{t, 2};
      end
    end
  end

  %% Console: results table
  fprintf('\n');
  fprintf('==========================================================================\n');
  fprintf('                           RESULTS TABLE\n');
  fprintf('==========================================================================\n');
  fprintf('%-6s | %-14s | %-14s | %-10s | %s\n', ...
    'Group', '# Perfect', 'Max Mean', 'Hull Area', 'Sim IDs');
  fprintf('--------------------------------------------------------------------------\n');

  for g = 1:n_groups
    ids_str = strjoin(results_table(g).sim_ids, ',');
    fprintf('%-6d | %5.1f +/- %5.1f | %6.1f +/- %5.1f | %10.4f | %s\n', ...
      g, results_table(g).n_perfect_mean, results_table(g).n_perfect_std, ...
      results_table(g).max_mean_rew, results_table(g).max_mean_rew_std, ...
      results_table(g).hull_area, ids_str);
    fprintf('        %s\n', config_labels{g});
  end
  fprintf('==========================================================================\n');

  %% ============================================================
  %% PLOT
  %% ============================================================

  %% Grid layout (shared by Figs 2, 3, 4A, 4B)
  n_cols = ceil(sqrt(n_groups));
  n_rows = ceil(n_groups / n_cols);
  g_margin_v = 0.1;  g_margin_h = 0.08;
  g_xsp = 0.05;      g_ysp = 0.05;
  g_w = (1 - 2*g_margin_h - (n_cols-1)*g_xsp) / n_cols;
  g_h = (1 - 2*g_margin_v - (n_rows-1)*g_ysp) / n_rows;

  wfig_g = round(interp1([3 18], [ 600 1800], n_groups, 'linear', 'extrap'));
  hfig_g = round(interp1([3 18], [ 510 1200], n_groups, 'linear', 'extrap'));

  box_labels = {};
  if (n_groups == 2) && opt.use_spine_labs
    box_labels = {'Spine', 'End-to-End'};
  end

  %% Figure 1: Overview traces + boxplots + results table
  %

  wfig = round(interp1([3 18], [1100 1620], n_groups, 'linear', 'extrap'));
  hfig = round(interp1([3 18], [400  480],  n_groups, 'linear', 'extrap'));


  %
  fig1 = figure('Name','Training Overview','Position',[1440 818 wfig hfig]);

  w1 = 0.3;
  w2 = 0.2;
  w3 = 0.45;
  xsp = 0.04;
  ysp_fig1 = 0.1;
  h = 0.35;

  % Panel 1: Solved episodes
  xs = xsp;
  ys = 0.95 - h;
  w = w1;
  pos = [xs,ys,w,h];
  ax = axes('Position',pos);
  ax.Color = 'none';
  hold on;

  for k = 1:n_sel
    g = group_idx(traces(k).model_idx);
    if strcmp(opt.abscissa,'Episode')
      x = traces(k).eps;
    else
      x = traces(k).step;
    end
    p = plot(x, traces(k).solved);
    p.Color = [cmap(g,:), 0.5];
  end
  ylabel('Perfect Episodes');
  yl = ylim;

  % Boxplot panel
  xs = xs + xsp + w;
  w = w2;
  pos = [xs,ys,w,h];
  ax_box = axes('Position',pos);

  box_groups = cell(n_groups, 1);
  for g = 1:n_groups
    box_groups{g} = metrics(g).solved_max(:);
  end
  plot_boxjitter(ax_box, box_groups, 'colors', cmap, ...
    'markersize', markersize_boxplot, 'labels', box_labels);
  ylabel(ax_box, 'Perfect Episodes');
  ylim(ax_box, yl);
  ax_box.Color = 'none';

  % Panel 2: Mean reward
  xs = xsp;
  w = w1;
  ys = ys - h - ysp_fig1;
  pos = [xs,ys,w,h];
  ax = axes('Position',pos);
  ax.Color = 'none';
  hold on;

  for k = 1:n_sel
    g = group_idx(traces(k).model_idx);
    if strcmp(opt.abscissa,'Episode')
      x = traces(k).eps;
    else
      x = double(traces(k).rew_step);
    end
    p = plot(x, traces(k).roll_mean);
    p.Color = [cmap(g,:), 0.5];
  end
  xlabel(opt.abscissa);
  ylabel(sprintf('Mean Ep Length (win=%d eps)', opt.mean_win));
  yl = ylim;

  % Boxplot panel
  xs = xs + xsp + w;
  w = w2;
  pos = [xs,ys,w,h];
  ax_box = axes('Position',pos);
  ax_box.Color = 'none';

  box_groups = cell(n_groups, 1);
  for g = 1:n_groups
    box_groups{g} = metrics(g).reward_max(:);
  end
  plot_boxjitter(ax_box, box_groups, 'colors', cmap, ...
    'markersize', markersize_boxplot, 'labels', box_labels);
  ylabel(ax_box, 'Max Mean Reward');
  ylim(ax_box, yl);

  % Steps to perfect boxplot
  xs = xs + xsp + w;
  w = w2;
  pos = [xs,ys,w,h];
  ax_box = axes('Position',pos);
  ax_box.Color = 'none';

  box_groups = cell(n_groups, 1);
  for g = 1:n_groups
    box_groups{g} = metrics(g).eps_to_perfect(:);
  end
  plot_boxjitter(ax_box, box_groups, 'colors', cmap, ...
    'markersize', markersize_boxplot, 'labels', box_labels);
  ylabel(ax_box, 'Episodes to Perfect');
  ylim(ax_box, [0 inf]);

  % Supra steps to perfect boxplot
  %xs = xs + xsp + w;
  %w = w2;
  %pos = [xs,ys,w,h];
  %ax_box = axes('Position',pos);
  %ax_box.Color = 'none';
  %box_groups = cell(n_groups, 1);
  %for g = 1:n_groups
  %  box_groups{g} = metrics(g).supra_steps_to_perfect(:);
  %end
  %plot_boxjitter(ax_box, box_groups, 'colors', cmap, ...
  %  'markersize', markersize_boxplot, 'labels', box_labels);
  %ylabel(ax_box, 'Supra Steps to Perfect');
  %ylim(ax_box, [0 inf]);

  %% Sim steps to perfect boxplot
  %xs = xs + xsp + w;
  %w = w2;
  %pos = [xs,ys,w,h];
  %ax_box = axes('Position',pos);
  %ax_box.Color = 'none';
  %box_groups = cell(n_groups, 1);
  %for g = 1:n_groups
  %  box_groups{g} = metrics(g).sim_steps_to_perfect(:);
  %end
  %plot_boxjitter(ax_box, box_groups, 'colors', cmap, ...
  %  'markersize', markersize_boxplot, 'labels', box_labels);
  %ylabel(ax_box, 'Sim Steps to Perfect');
  %ylim(ax_box, [0 inf]);

  fprintf('\nLegend:\n');
  for g = 1:n_groups
    fprintf('  [%d] %s\n', g, config_labels{g});
  end

  % Results table (embedded in Figure 1, right column)
  extra_fields = opt.table_fields;
  if ischar(extra_fields) && strcmp(extra_fields, 'all')
    % All custom param fields = everything not in the standard metric set
    builtin_fields = {'group','config','label','sim_ids','model_ids', ...
      'n_perfect_mean','n_perfect_std','max_mean_rew','max_mean_rew_std', ...
      'hull_area', ...
      'eps_to_perfect_mean','eps_to_perfect_std', ...
      'supra_steps_to_perfect_mean','supra_steps_to_perfect_std', ...
      'sim_steps_to_perfect_mean','sim_steps_to_perfect_std'};
    all_fns = fieldnames(results_table);
    extra_fields = setdiff(all_fns, builtin_fields, 'stable')';
  end
  extra_hdrs = {};
  for ef = 1:numel(extra_fields)
    fname = extra_fields{ef};
    % Use last segment of underscore-separated name as short label
    parts = strsplit(fname, '_');
    extra_hdrs{ef} = parts{end};
  end

  % Check if spine type differs across groups
  has_spine_col = any(contains(config_labels, 'spine='));

  if has_spine_col
      header = sprintf('%-4s  %-18s  %-14s  %-12s  %-10s  %-10s  %-8s', ...
          'Grp', 'Spine', '# Perfect', 'Max Mean', 'Eps->P', 'Sim->P(k)', 'Hull');
  else
      header = sprintf('%-4s  %-14s  %-12s  %-10s  %-10s  %-8s', ...
          'Grp', '# Perfect', 'Max Mean', 'Eps->P', 'Sim->P(k)', 'Hull');
  end


  for ef = 1:numel(extra_hdrs)
    header = [header, sprintf('  %-8s', extra_hdrs{ef})];
  end
  sep = repmat('-', 1, numel(header));
  lines_str = {header, sep};

  [~,sort_idx] = sort([results_table.n_perfect_mean],'descend');

  for iq = 1:numel(sort_idx)
    g = sort_idx(iq);
    if has_spine_col
      % Extract spine type from config key (first token: spine=X)
      tok_g = strsplit(unique_configs{g}, ', ');
      spine_tok = tok_g{1};  % 'spine=identity'
      spine_val = spine_tok(7:end);  % strip 'spine='
      row_str = sprintf('%-4d  %-18s  %5.1f ±%-4.1f  %6.1f ±%-4.1f  %5.0f ±%-4.0f  %5.0f ±%-4.0f  %6.4f', ...
          g, spine_val, ...
          results_table(g).n_perfect_mean,                results_table(g).n_perfect_std, ...
          results_table(g).max_mean_rew,                  results_table(g).max_mean_rew_std, ...
          results_table(g).eps_to_perfect_mean,           results_table(g).eps_to_perfect_std, ...
          results_table(g).sim_steps_to_perfect_mean/1e3, results_table(g).sim_steps_to_perfect_std/1e3, ...
          results_table(g).hull_area);
    else
      row_str = sprintf('%-4d  %5.1f ±%-4.1f  %6.1f ±%-4.1f  %5.0f ±%-4.0f  %5.0f ±%-4.0f  %6.4f', ...
        g, results_table(g).n_perfect_mean,                results_table(g).n_perfect_std, ...
        results_table(g).max_mean_rew,                     results_table(g).max_mean_rew_std, ...
        results_table(g).eps_to_perfect_mean,              results_table(g).eps_to_perfect_std, ...
        results_table(g).sim_steps_to_perfect_mean/1e3,    results_table(g).sim_steps_to_perfect_std/1e3, ...
        results_table(g).hull_area);
    end
    for ef = 1:numel(extra_fields)
      fname = extra_fields{ef};
      % Suffix match: 'damping' finds 'env_kwargs_damping'
      if ~isfield(results_table, fname)
        fns = fieldnames(results_table);
        match = fns(endsWith(fns, fname));
        if ~isempty(match), fname = match{1}; end
      end

      if isfield(results_table, fname)
          val = results_table(g).(fname);
          if isnumeric(val) && numel(val) > 1
              vs = strjoin(arrayfun(@(x) sprintf('%g', x), val, 'Uni', false), ',');
              row_str = [row_str, sprintf('  [%s]', vs)];
          elseif isnumeric(val)
              row_str = [row_str, sprintf('  %8g', val)];
          else
              row_str = [row_str, sprintf('  %-8s', string(val))];
          end
      else
          row_str = [row_str, sprintf('  %-8s', '--')];
      end
    end
    lines_str{end+1} = row_str;
  end

  table_text = strjoin(lines_str, '\n');

  table_xs = 3*xsp + w1 + w2;
  annotation(fig1, 'textbox', [table_xs 0.05 w3 0.9], ...
    'String', table_text, ...
    'FontName', 'FixedWidth', ...
    'FontSize', 7, ...
    'EdgeColor', [0.7 0.7 0.7], ...
    'FitBoxToText', 'on', ...
    'Interpreter', 'none');

  %ann = annotation(fig1, 'textbox', [0 0.01 1 1]);
  %ann.String = log_root;
  %ann.EdgeColor = 'none';
  %ann.Interpreter = 'none';
  %ann.HorizontalAlignment = 'right';
  %ann.VerticalAlignment = 'bottom';

  %% Figure 2: IC Sensitivity
  if ~opt.fig1_only
  fig2 = figure('Name','IC-Sensitivity','Position',[100 329 wfig_g hfig_g]);

  for g = 1:n_groups
    col = mod(g-1, n_cols) + 1;
    row = ceil(g / n_cols);
    xs = g_margin_h + (col-1)*(g_w + g_xsp);
    ys = 1 - g_margin_v - row*g_h - (row-1)*g_ysp;
    pos = [xs, ys, g_w, g_h];
    ax = axes('Position', pos);

    g_idx = all_group == g;
    g_q0 = all_q0(g_idx);
    g_q1 = all_q1(g_idx);
    g_rew = all_rew(g_idx);

    g_perfect  = all_perfect(g_idx);
    g_rew_norm = g_rew / rew_lim(2);

    if strcmp(opt.ic_type, 'pend_only')
      g_pend = all_pend(g_idx);
      plot_pend_ics(ax, g_pend, g_rew_norm, 'perfect_mask', g_perfect);
    else
      plot_ic_surface(ax, g_q0, g_q1, g_rew, ...
        'perfect_mask', g_perfect, ...
        'rew_max', max(g_rew), ...
        'do_kernel', opt.do_kernel, ...
        'markersize', markersize, ...
        'view_angle', isoview, ...
        'q0_lim', q0_lim, ...
        'q1_lim', q1_lim);
    end

    % Annotation
    mean_rew_norm = mean(g_rew_norm);

    pec_w = 0.65;
    pec_h = 0.85;
    pos_ann = [pos(1) + pec_w*g_w, pos(2)+pec_h*g_h, ...
               (1-pec_w)*g_w, (1-pec_h)*g_h];
    str = {['[',num2str(g),'] mean: ', num2str(mean_rew_norm,'%.2f')], ...
           ['solved: ',num2str(sum(g_perfect))]};
    ann = annotation('textbox', 'Position', pos_ann);
    ann.String = str;
    ann.FontSize = 10;
    ann.EdgeColor = 'none';
    ann.BackgroundColor = 'k';
    ann.Color = 'w';
    ann.VerticalAlignment = 'top';
    ann.HorizontalAlignment = 'right';

    is_bottom = row == n_rows;
    is_left = col == 1;
    if is_bottom && is_left
      if strcmp(opt.ic_type, 'pend_only')
        xlabel('x'); ylabel('y');
      else
        xlabel('q0'); ylabel('q1');
      end
    end

    last_pos = pos;
  end
  colormap(c1map);
  clim([0 1]);

  %% Figure 3: Perfect IC Arms
  fig3 = figure('Name','Perfect IC-Configurations','Position',[100 50 wfig_g hfig_g]);

  for g = 1:n_groups
    col = mod(g-1, n_cols) + 1;
    row = ceil(g / n_cols);
    xs = g_margin_h + (col-1)*(g_w + g_xsp);
    ys = 1 - g_margin_v - row*g_h - (row-1)*g_ysp;
    pos = [xs, ys, g_w, g_h];
    ax = axes('Position', pos);

    g_idx      = all_group == g;
    g_q0       = all_q0(g_idx);
    g_q1       = all_q1(g_idx);
    g_rew      = all_rew(g_idx);
    g_rew_norm = g_rew / rew_lim(2);
    is_perfect = all_perfect(g_idx);

    if strcmp(opt.ic_type, 'pend_only')
      g_pend = all_pend(g_idx);
      plot_pend_ics(ax, g_pend(is_perfect), ones(sum(is_perfect),1), ...
        'perfect_mask', true(sum(is_perfect),1), 'alpha', 0.8);
    else
      plot_arm_configs(ax, g_q0(is_perfect), g_q1(is_perfect), ...
        g_rew_norm(is_perfect), 'L', L);
    end

    pos_ann = [pos(1) + 0.65*g_w, pos(2) + 0.85*g_h, 0.35*g_w, 0.15*g_h];
    ann = annotation('textbox', 'Position', pos_ann);
    ann.String = sprintf('[%d] %d/%d', g, sum(is_perfect),numel(g_q0));
    ann.FontSize = 10;
    ann.EdgeColor = 'none';
    ann.BackgroundColor = 'none';
    ann.Color = 'k';
    ann.VerticalAlignment = 'top';
    ann.HorizontalAlignment = 'right';

    is_bottom = row == n_rows;
    is_left = col == 1;
    if is_bottom && is_left
      xlabel('x'); ylabel('y');
    end
  end
  colormap([0 0 0]);
  clim([0 1]);

  %cb_pos = [pos(1) + g_w + 0.25*g_xsp, pos(2), 0.02, g_h];
  %cb_ax = axes('Position', cb_pos, 'Visible', 'off');
  %colorbar(cb_ax, 'Position', cb_pos);
  %clim([0 1]);

  %% Figure 4A: IC band arms
  fig4a = figure('Name','IC rew > threshold','Position',[100 350 wfig_g hfig_g]);

  for g = 1:n_groups
    col = mod(g-1, n_cols) + 1;
    row = ceil(g / n_cols);
    xs = g_margin_h + (col-1)*(g_w + g_xsp);
    ys = 1 - g_margin_v - row*g_h - (row-1)*g_ysp;
    pos = [xs, ys, g_w, g_h];
    ax = axes('Position', pos);

    plot_arm_configs(ax, band_ic(g).q0, band_ic(g).q1, band_ic(g).rew, ...
      'L', L, 'alpha', 0.1);

    pos_ann = [pos(1) + 0.65*g_w, pos(2) + 0.85*g_h, 0.35*g_w, 0.15*g_h];
    ann = annotation('textbox', 'Position', pos_ann);
    ann.String = sprintf('[%d] n=%d', g, numel(band_ic(g).q0));
    ann.FontSize = 10;
    ann.EdgeColor = 'none';
    ann.BackgroundColor = 'none';
    ann.Color = 'k';
    ann.VerticalAlignment = 'top';
    ann.HorizontalAlignment = 'right';

    if row == n_rows && col == 1
      xlabel('x'); ylabel('y');
    end
  end
  colormap(c1map);
  clim([0 1]);

  cb_pos = [pos(1) + g_w + 0.25*g_xsp, pos(2), 0.02, g_h];
  cb_ax = axes('Position', cb_pos, 'Visible', 'off');
  colorbar(cb_ax, 'Position', cb_pos);
  clim([0 1]);

  %% Figure 4B: FC band arms
  fig4b = figure('Name','FC rew < perfect','Position',[100 350 wfig_g hfig_g]);

  for g = 1:n_groups
    col = mod(g-1, n_cols) + 1;
    row = ceil(g / n_cols);
    xs = g_margin_h + (col-1)*(g_w + g_xsp);
    ys = 1 - g_margin_v - row*g_h - (row-1)*g_ysp;
    pos = [xs, ys, g_w, g_h];
    ax = axes('Position', pos);

    plot_arm_configs(ax, band_fc(g).q0, band_fc(g).q1, band_fc(g).rew, ...
      'L', L, 'alpha', 0.1);

    pos_ann = [pos(1) + 0.65*g_w, pos(2) + 0.85*g_h, 0.35*g_w, 0.15*g_h];
    ann = annotation('textbox', 'Position', pos_ann);
    ann.String = sprintf('[%d] n=%d', g, numel(band_fc(g).q0));
    ann.FontSize = 10;
    ann.EdgeColor = 'none';
    ann.BackgroundColor = 'none';
    ann.Color = 'k';
    ann.VerticalAlignment = 'top';
    ann.HorizontalAlignment = 'right';

    if row == n_rows && col == 1
      xlabel('x'); ylabel('y');
    end
  end
  colormap(c1map);
  clim([0 1]);

  cb_pos = [pos(1) + g_w + 0.25*g_xsp, pos(2), 0.02, g_h];
  cb_ax = axes('Position', cb_pos, 'Visible', 'off');
  colorbar(cb_ax, 'Position', cb_pos);
  clim([0 1]);
  end

  %% Figure 5: Eval — mean per group vs checkpoint
  if ~opt.fig1_only && opt.eval

  fig5 = figure('Name','Eval','Position',[100 100 wfig_g hfig_g]);
  xsp5 = 0.1; ysp5 = 0.08; lm = 0.08; rm = 0.07; bm = 0.1; tm = 0.05;
  ph = (1 - bm - tm - ysp5) / 2;
  pw_trace = 0.52;
  xsp_box  = 0.05;
  w_box    = 0.28;

  ax_np   = axes('Position', [lm, bm + ph + ysp5, pw_trace, ph]);
  ax_mrew = axes('Position', [lm, bm,             pw_trace, ph]);
  hold(ax_np,   'on'); ax_np.Color   = 'none';
  hold(ax_mrew, 'on'); ax_mrew.Color = 'none';

  ax_box_np   = axes('Position', [lm + pw_trace + xsp_box, bm + ph + ysp5, w_box, ph]);
  ax_box_mrew = axes('Position', [lm + pw_trace + xsp_box, bm,             w_box, ph]);
  ax_box_np.Color = 'none'; ax_box_mrew.Color = 'none';

  for g = 1:n_groups
    % Collect eval traces for selected models in this group
    np_mat   = [];
    mrew_mat = [];
    x_ref    = [];

    for k = 1:n_sel
      if group_idx(traces(k).model_idx) ~= g, continue; end
      if isempty(traces(k).eval_step), continue; end
      n = numel(traces(k).eval_step);
      if isempty(x_ref) || n < numel(x_ref)
        x_ref = double(traces(k).eval_step(1:n));
      end
      np_mat   = [np_mat,   traces(k).eval_n_perfect(1:n)];
      mrew_mat = [mrew_mat, traces(k).eval_mean_rew(1:n)];
    end

    if isempty(np_mat), continue; end
    n_min = min(size(np_mat,1), numel(x_ref));
    x   = (1:n_min)';
    np  = np_mat(1:n_min, :);
    mr  = mrew_mat(1:n_min, :);

    % n_perfect panel
    mu  = mean(np,  2);
    sd  = std(np,   0, 2);
    xf  = [x; flipud(x)];
    %fill(ax_np, xf, [mu+sd; flipud(mu-sd)], cmap(g,:), ...
    %  'FaceAlpha', 0.5, 'EdgeColor', 'none');
    plot(ax_np, x, mu, 'Color', cmap(g,:), 'LineWidth', 1.5);

    % mean_rew panel
    mu  = mean(mr,  2);
    sd  = std(mr,   0, 2);
    %fill(ax_mrew, xf, [mu+sd; flipud(mu-sd)], cmap(g,:), ...
    %  'FaceAlpha', 0.5, 'EdgeColor', 'none');
    plot(ax_mrew, x, mu, 'Color', cmap(g,:), 'LineWidth', 1.5);
  end

  ylabel(ax_np,   'Eval n\_perfect');
  ylabel(ax_mrew, 'Eval mean rew');
  xlabel(ax_mrew, 'Checkpoint');
  ax_np.XTickLabel = {};

  all_np_max   = cellfun(@(x) max(x(:)), {metrics.eval_n_perfect_max}, 'UniformOutput', false);
  all_mrew_max = cellfun(@(x) max(x(:)), {metrics.eval_mean_rew_max},  'UniformOutput', false);
  all_np_max(cellfun(@isempty, all_np_max))     = {0};
  all_mrew_max(cellfun(@isempty, all_mrew_max)) = {0};
  yl_np   = [0, max(cell2mat(all_np_max))];
  yl_mrew = [0, max(cell2mat(all_mrew_max))];
  ylim(ax_np,   yl_np);
  ylim(ax_mrew, yl_mrew);

  box_groups = cell(n_groups, 1);
  for g = 1:n_groups
    box_groups{g} = metrics(g).eval_n_perfect_max(:);
  end
  plot_boxjitter(ax_box_np, box_groups, 'colors', cmap, ...
    'markersize', markersize_boxplot, 'labels', box_labels);
  ylabel(ax_box_np, 'Max Eval Perfect');
  ylim(ax_box_np, yl_np);

  box_groups = cell(n_groups, 1);
  for g = 1:n_groups
    box_groups{g} = metrics(g).eval_mean_rew_max(:);
  end
  plot_boxjitter(ax_box_mrew, box_groups, 'colors', cmap, ...
    'markersize', markersize_boxplot, 'labels', box_labels);
  ylabel(ax_box_mrew, 'Max Eval Mean Rew');
  ylim(ax_box_mrew, yl_mrew);

  end

  %% Save results

  save(fullfile(savepath, 'results_table.mat'), 'results_table');

  if ~opt.fig1_only
    figs = {fig1, fig2, fig3, fig4a, fig4b};
    if opt.eval, figs{end+1} = fig5; end
  else
    figs = {fig1};
  end
  png_paths = {};
  for f = 1:numel(figs)
    ann = annotation(figs{f}, 'textbox', [0 0.01 1 1]);
    ann.String = log_root;
    ann.EdgeColor = 'none';
    ann.Interpreter = 'none';
    ann.HorizontalAlignment = 'right';
    ann.VerticalAlignment = 'bottom';

    fname = matlab.lang.makeValidName(figs{f}.Name);
    fpath = fullfile(savepath, [fname '.png']);
    print(figs{f}, '-dpng', fpath);
    png_paths{end+1} = struct('name', figs{f}.Name, 'path', fpath);
  end

  if opt.html_report
    html_fid = fopen(fullfile(savepath, 'report.html'), 'w');
    fprintf(html_fid, ['<!DOCTYPE html><html><head><meta charset="utf-8">' ...
      '<title>%s</title><style>' ...
      'body{font-family:monospace;max-width:1400px;margin:0 auto;padding:20px;background:#fff}' ...
      'h1{font-size:13px;color:#888;margin-bottom:8px}' ...
      'h2{font-size:12px;margin-top:30px;margin-bottom:6px}' ...
      'img{width:auto;height:auto;border:1px solid #ddd;display:block;margin-bottom:4px}' ...
      'table{border-collapse:collapse;font-size:11px;margin-bottom:20px}' ...
      'th,td{border:1px solid #ccc;padding:3px 8px;text-align:left}' ...
      'th{background:#f5f5f5}' ...
      '</style></head><body>\n'], log_root);
    fprintf(html_fid, '<h1>%s &nbsp;&middot;&nbsp; %s</h1>\n', ...
      log_root, datestr(now));

    % Results table
    fprintf(html_fid, '<h2>Results</h2>\n');
    fprintf(html_fid, ['<table><tr><th>Label</th><th>#Perfect (mean&pm;std)</th>' ...
      '<th>MaxMean (mean&pm;std)</th><th>HullArea</th>' ...
      '<th>EpsToPerf (mean&pm;std)</th></tr>\n']);
    for g = 1:numel(results_table)
      r = results_table(g);
      fprintf(html_fid, '<tr><td>%s</td><td>%.1f&pm;%.1f</td><td>%.1f&pm;%.1f</td><td>%.4f</td><td>%.0f&pm;%.0f</td></tr>\n', ...
        r.label, r.n_perfect_mean, r.n_perfect_std, ...
        r.max_mean_rew, r.max_mean_rew_std, ...
        r.hull_area, r.eps_to_perfect_mean, r.eps_to_perfect_std);
    end
    fprintf(html_fid, '</table>\n');

    % Figures
    for f = 1:numel(png_paths)
      p = png_paths{f};
      img_fid = fopen(p.path, 'rb');
      bytes = fread(img_fid, inf, 'uint8=>uint8'); fclose(img_fid);
      b64 = matlab.net.base64encode(bytes);
      fprintf(html_fid, '<h2>%s</h2>\n<img src="data:image/png;base64,%s">\n', ...
        p.name, b64);
    end

    fprintf(html_fid, '</body></html>\n');
    fclose(html_fid);
  end

  fprintf('Saved to: %s\n', savepath);

  %% Restore default
  set(0, 'DefaultFigureVisible', 'on');
  toc
  diary off
end
