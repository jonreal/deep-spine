
% Two-model comparison with seed pairing
% Compares identity (end-to-end) vs spine (ia_reflex)
% Connecting lines show matched seeds

paths = {
  '../../experiments/hebbian/logs/PNAS/rl-baseline3/df.mat'}

% step | Episode
abscissa = 'step';

%% Tunable parameters
mean_win = 100;  % rolling window for mean reward

%% Load data
T = table();
for i=1:numel(paths)
  S = load(paths{i});
  S.table.Step = S.table.Step(:);
  S.table.Value = S.table.Value(:);
  S.table.Tag = S.table.Tag(:);
  S.table.ModelID = S.table.ModelID(:);
  S.table.Model = S.table.Model(:);
  Ti = struct2table(S.table);
  T = [T; Ti];
end

all_models = unique(T.ModelID);
n_models = numel(all_models);

%% Extract seed and model type for each model
model_seeds = zeros(n_models, 1);
model_types = cell(n_models, 1);  % 'identity' or 'spine'

for i = 1:n_models
    % Get model type from data
    iq = strcmp(T.ModelID, all_models{i});
    model_types{i} = unique(T.Model(iq));
    model_types{i} = model_types{i}{1};  % extract string from cell

    % Extract seed from model ID (e.g., '...-seed_1' or just parse order)
    parts = split(all_models{i}, '-');
    seed_idx = find(startsWith(parts, 'seed_'));
    if ~isempty(seed_idx)
        seed_str = parts{seed_idx};
        model_seeds(i) = str2double(extractAfter(seed_str, 'seed_'));
    else
        % Fallback: use model index within type as seed proxy
        model_seeds(i) = i;
    end
end

%% Separate by model type and sort by seed
identity_idx = find(strcmp(model_types, 'identity'));
spine_idx = find(~strcmp(model_types, 'identity'));

[~, sort_id] = sort(model_seeds(identity_idx));
identity_idx = identity_idx(sort_id);

[~, sort_sp] = sort(model_seeds(spine_idx));
spine_idx = spine_idx(sort_sp);

n_seeds = min(numel(identity_idx), numel(spine_idx));

fprintf('Found %d identity models, %d spine models, %d seed pairs\n', ...
    numel(identity_idx), numel(spine_idx), n_seeds);

%% Colors
col_id = [0 0 0];      % black for identity/end-to-end
col_sp = [1 0 0];      % red for spine
col_line = [0.5 0.5 0.5];  % gray for connecting lines

%% Figure setup
%figure('Position',[1440 818 556 300]);
figure('Position',[1440 969 484 149]);

w1 = 0.50;
w2 = 0.25;
xsp = 0.075;
ysp = 0.15;
h = 0.65;

%% Panel 1: # Perfect Episodes
xs = 0.1;
ys = 0.95 - h;
w = w1;

pos = [xs,ys,w,h];
ax = axes('Position',pos);
hold on;

bl = zeros(n_seeds, 1);  % identity metrics
sp = zeros(n_seeds, 1);  % spine metrics

% Plot learning curves and collect metrics
for s = 1:n_seeds
    % Identity model
    i = identity_idx(s);
    iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/solved');
    [step, idx] = sort(T.Step(iq));
    solved = T.Value(iq);
    solved = solved(idx);

    iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/eps');
    [~, idx] = sort(T.Step(iq));
    eps_ = T.Value(iq);
    eps_ = eps_(idx);

    if strcmp(abscissa,'Episode')
        x = eps_;
    else
        x = step;
    end

    p = plot(x, solved);
    p.Color = [col_id, 0.5];
    bl(s) = max(solved);

    % Spine model
    i = spine_idx(s);
    iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/solved');
    [step, idx] = sort(T.Step(iq));
    solved = T.Value(iq);
    solved = solved(idx);

    iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/eps');
    [~, idx] = sort(T.Step(iq));
    eps_ = T.Value(iq);
    eps_ = eps_(idx);

    if strcmp(abscissa,'Episode')
        x = eps_;
    else
        x = step;
    end

    p = plot(x, solved);
    p.Color = [col_sp, 0.5];
    sp(s) = max(solved);
end
ylabel('Perfect Episodes');
xlabel(abscissa);
yl = ylim;

% Boxplot panel with seed pairing
xs = xs + xsp + w;
w = w2;
pos = [xs,ys,w,h];
ax = axes('Position',pos);
hold on;

% Deterministic jitter (evenly spaced, inward facing)
jitter = 0.15 * (1:n_seeds) / n_seeds;

x_bl = 1.25 + jitter;  % identity points go right (toward middle)
x_sp = 1.75 - jitter;  % spine points go left (toward middle)

% Draw connecting lines first (behind points)
for s = 1:n_seeds
    plot([x_bl(s), x_sp(s)], [bl(s), sp(s)], '-', ...
        'Color', [col_line, 0.4], 'LineWidth', 0.5);
end

% Scatter points
scatter(x_bl, bl, 36, col_id, 'filled', 'MarkerFaceAlpha', 0.7);
scatter(x_sp, sp, 36, col_sp, 'filled', 'MarkerFaceAlpha', 0.7);

% Boxplots
boxplot([bl(:), sp(:)], 'Widths', 0.1, 'Colors', [col_id; col_sp]);
xticklabels({'end-to-end', 'spine'});
ylabel({'Number of Perfect', 'Episodes'});
xlim([0.5 2.5]);
ylim(yl);

%% Figure 2: IC Sensitivity side-by-side
figure('Position',[100 100 600 300]);

n_last_eps = 300;  % only use last N episodes
success_thresh = 1400;

% Collect IC data for identity models
id_q0 = []; id_q1 = []; id_rew = [];
for s = 1:n_seeds
    i = identity_idx(s);

    iq_rew = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ep_rew');
    [~, idx_rew] = sort(T.Step(iq_rew));
    rew = T.Value(iq_rew); rew = rew(idx_rew);

    iq_q0 = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ic_q0');
    [~, idx_q0] = sort(T.Step(iq_q0));
    q0 = T.Value(iq_q0); q0 = q0(idx_q0);

    iq_q1 = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ic_q1');
    [~, idx_q1] = sort(T.Step(iq_q1));
    q1 = T.Value(iq_q1); q1 = q1(idx_q1);

    n = min([numel(rew), numel(q0), numel(q1)]);
    n_take = min(n_last_eps, n);
    start_idx = n - n_take + 1;

    id_q0 = [id_q0; q0(start_idx:n)];
    id_q1 = [id_q1; q1(start_idx:n)];
    id_rew = [id_rew; rew(start_idx:n)];
end

% Collect IC data for spine models
sp_q0 = []; sp_q1 = []; sp_rew = [];
for s = 1:n_seeds
    i = spine_idx(s);

    iq_rew = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ep_rew');
    [~, idx_rew] = sort(T.Step(iq_rew));
    rew = T.Value(iq_rew); rew = rew(idx_rew);

    iq_q0 = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ic_q0');
    [~, idx_q0] = sort(T.Step(iq_q0));
    q0 = T.Value(iq_q0); q0 = q0(idx_q0);

    iq_q1 = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ic_q1');
    [~, idx_q1] = sort(T.Step(iq_q1));
    q1 = T.Value(iq_q1); q1 = q1(idx_q1);

    n = min([numel(rew), numel(q0), numel(q1)]);
    n_take = min(n_last_eps, n);
    start_idx = n - n_take + 1;

    sp_q0 = [sp_q0; q0(start_idx:n)];
    sp_q1 = [sp_q1; q1(start_idx:n)];
    sp_rew = [sp_rew; rew(start_idx:n)];
end

% Common axis limits
all_q0 = [id_q0; sp_q0];
all_q1 = [id_q1; sp_q1];
q0_lim = [min(all_q0), max(all_q0)];
q1_lim = [min(all_q1), max(all_q1)];

% Left: Identity (end-to-end)
subplot(1,2,1);
hold on;
fail_idx = id_rew < success_thresh;
scatter(id_q0(fail_idx), id_q1(fail_idx), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.2);
scatter(id_q0(~fail_idx), id_q1(~fail_idx), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
xlim(q0_lim); ylim(q1_lim);
xlabel('q0'); ylabel('q1');
success_rate = 100 * sum(~fail_idx) / numel(id_rew);
title(sprintf('End-to-end: %.0f%% success', success_rate));

% Right: Spine
subplot(1,2,2);
hold on;
fail_idx = sp_rew < success_thresh;
scatter(sp_q0(fail_idx), sp_q1(fail_idx), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.2);
scatter(sp_q0(~fail_idx), sp_q1(~fail_idx), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
xlim(q0_lim); ylim(q1_lim);
xlabel('q0'); ylabel('q1');
success_rate = 100 * sum(~fail_idx) / numel(sp_rew);
title(sprintf('Spine: %.0f%% success', success_rate));

sgtitle(sprintf('IC Sensitivity (last %d eps)', n_last_eps));

%% Figure 3: Verify IC sequence matches for same seed
figure('Position',[100 400 800 300]);

% Compare IC sequences for each seed pair
for s = 1:n_seeds
    % Identity model ICs
    i = identity_idx(s);
    iq_q0 = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ic_q0');
    [step_id, idx] = sort(T.Step(iq_q0));
    q0_id = T.Value(iq_q0); q0_id = q0_id(idx);

    iq_q1 = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ic_q1');
    [~, idx] = sort(T.Step(iq_q1));
    q1_id = T.Value(iq_q1); q1_id = q1_id(idx);

    % Spine model ICs
    i = spine_idx(s);
    iq_q0 = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ic_q0');
    [step_sp, idx] = sort(T.Step(iq_q0));
    q0_sp = T.Value(iq_q0); q0_sp = q0_sp(idx);

    iq_q1 = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ic_q1');
    [~, idx] = sort(T.Step(iq_q1));
    q1_sp = T.Value(iq_q1); q1_sp = q1_sp(idx);

    % Align lengths
    n = min([numel(q0_id), numel(q0_sp)]);

    % Subplot: scatter identity IC vs spine IC
    subplot(1, n_seeds, s);
    hold on;
    scatter(q0_id(1:n), q0_sp(1:n), 5, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
    scatter(q1_id(1:n), q1_sp(1:n), 5, 'r', 'filled', 'MarkerFaceAlpha', 0.3);

    % Diagonal line (perfect match)
    lim = [min([q0_id; q1_id; q0_sp; q1_sp]), max([q0_id; q1_id; q0_sp; q1_sp])];
    plot(lim, lim, 'k--', 'LineWidth', 1);

    xlim(lim); ylim(lim);
    xlabel('Identity IC');
    ylabel('Spine IC');
    title(sprintf('Seed %d', s));
    axis square;

    % Compute correlation
    r_q0 = corr(q0_id(1:n), q0_sp(1:n));
    r_q1 = corr(q1_id(1:n), q1_sp(1:n));
    text(0.05, 0.95, sprintf('r=%.3f', min(r_q0, r_q1)), ...
        'Units', 'normalized', 'FontSize', 8, 'VerticalAlignment', 'top');
end
sgtitle('IC Sequence Match (blue=q0, red=q1, should be on diagonal)');

return

%% Panel 2: Mean Reward
xs = 0.1;
w = w1;
ys = ys - h - ysp;

pos = [xs,ys,w,h];
ax = axes('Position',pos);
hold on;

bl = zeros(n_seeds, 1);
sp = zeros(n_seeds, 1);

for s = 1:n_seeds
    % Identity model
    i = identity_idx(s);
    iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ep_rew');
    [step, idx] = sort(T.Step(iq));
    val = T.Value(iq);
    val = val(idx);
    roll_mean = movmean(val, mean_win);

    iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/eps');
    [~, idx] = sort(T.Step(iq));
    eps_ = T.Value(iq);
    eps_ = eps_(idx);

    if strcmp(abscissa,'Episode')
        x = eps_;
    else
        x = step;
    end

    p = plot(x, roll_mean);
    p.Color = [col_id, 0.5];
    bl(s) = max(roll_mean);

    % Spine model
    i = spine_idx(s);
    iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ep_rew');
    [step, idx] = sort(T.Step(iq));
    val = T.Value(iq);
    val = val(idx);
    roll_mean = movmean(val, mean_win);

    iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/eps');
    [~, idx] = sort(T.Step(iq));
    eps_ = T.Value(iq);
    eps_ = eps_(idx);

    if strcmp(abscissa,'Episode')
        x = eps_;
    else
        x = step;
    end

    p = plot(x, roll_mean);
    p.Color = [col_sp, 0.5];
    sp(s) = max(roll_mean);
end
xlabel(abscissa);
ylabel(sprintf('Mean Reward (win=%d)', mean_win));

% Boxplot panel with seed pairing
xs = xs + xsp + w;
w = w2;
pos = [xs,ys,w,h];
ax = axes('Position',pos);
hold on;

% Same jitter as Panel 1
x_bl = 1.25 + jitter;
x_sp = 1.75 - jitter;

% Draw connecting lines
for s = 1:n_seeds
    plot([x_bl(s), x_sp(s)], [bl(s), sp(s)], '-', ...
        'Color', [col_line, 0.4], 'LineWidth', 0.5);
end

% Scatter points
scatter(x_bl, bl, 36, col_id, 'filled', 'MarkerFaceAlpha', 0.7);
scatter(x_sp, sp, 36, col_sp, 'filled', 'MarkerFaceAlpha', 0.7);

% Boxplots
boxplot([bl(:), sp(:)], 'Widths', 0.1, 'Colors', [col_id; col_sp]);
xticklabels({'end-to-end', 'spine'});
ylabel({'Mean', 'Reward'});
xlim([0.9 2.1]);

%% Print seed pairing info
fprintf('\nSeed pairing:\n');
for s = 1:n_seeds
    fprintf('  Pair %d: %s vs %s\n', s, all_models{identity_idx(s)}, all_models{spine_idx(s)});
end
