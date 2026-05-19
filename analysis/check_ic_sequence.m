function check_ic_sequence(path)
% CHECK_IC_SEQUENCE  Verify IC sequences match across same-seed runs.
%   check_ic_sequence(path)  loads a df.mat and plots identity-IC vs
%   spine-IC for each seed pair. Points should fall on the diagonal if
%   seeding is consistent.

Step    = h5read(path, '/table/Step');
Value   = h5read(path, '/table/Value');
Tag     = cellstr(h5read(path, '/table/Tag'));
ModelID = cellstr(h5read(path, '/table/ModelID'));
Model   = cellstr(h5read(path, '/table/Model'));
T = table(Step(:), Value(:), Tag(:), ModelID(:), Model(:), ...
  'VariableNames', {'Step','Value','Tag','ModelID','Model'});

all_models = unique(T.ModelID);
n_models   = numel(all_models);

%% Extract seed and model type
model_seeds = zeros(n_models, 1);
model_types = cell(n_models, 1);

for i = 1:n_models
  iq = strcmp(T.ModelID, all_models{i});
  mt = unique(T.Model(iq));
  model_types{i} = mt{1};

  parts    = split(all_models{i}, '-');
  seed_idx = find(startsWith(parts, 'seed_'));
  if ~isempty(seed_idx)
    model_seeds(i) = str2double(extractAfter(parts{seed_idx}, 'seed_'));
  else
    model_seeds(i) = i;
  end
end

%% Group by model type and sort by seed
identity_idx = find(strcmp(model_types, 'identity'));
spine_idx    = find(~strcmp(model_types, 'identity'));

[~, s] = sort(model_seeds(identity_idx)); identity_idx = identity_idx(s);
[~, s] = sort(model_seeds(spine_idx));    spine_idx    = spine_idx(s);

n_seeds = min(numel(identity_idx), numel(spine_idx));
fprintf('Found %d identity, %d spine, %d seed pairs\n', ...
  numel(identity_idx), numel(spine_idx), n_seeds);

%% Plot
figure('Name', 'IC Sequence Check', 'Position', [100 400 300*n_seeds 300]);

for s = 1:n_seeds
  % Identity ICs
  i  = identity_idx(s);
  iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag, 'eps/ic_q0');
  [~, idx] = sort(T.Step(iq)); q0_id = T.Value(iq); q0_id = q0_id(idx);

  iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag, 'eps/ic_q1');
  [~, idx] = sort(T.Step(iq)); q1_id = T.Value(iq); q1_id = q1_id(idx);

  % Spine ICs
  i  = spine_idx(s);
  iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag, 'eps/ic_q0');
  [~, idx] = sort(T.Step(iq)); q0_sp = T.Value(iq); q0_sp = q0_sp(idx);

  iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag, 'eps/ic_q1');
  [~, idx] = sort(T.Step(iq)); q1_sp = T.Value(iq); q1_sp = q1_sp(idx);

  n = min([numel(q0_id), numel(q0_sp)]);

  subplot(1, n_seeds, s);
  hold on;
  scatter(q0_id(1:n), q0_sp(1:n), 5, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
  scatter(q1_id(1:n), q1_sp(1:n), 5, 'r', 'filled', 'MarkerFaceAlpha', 0.3);

  lim = [min([q0_id; q1_id; q0_sp; q1_sp]), max([q0_id; q1_id; q0_sp; q1_sp])];
  plot(lim, lim, 'k--', 'LineWidth', 1);
  xlim(lim); ylim(lim); axis square;

  r_q0 = corr(q0_id(1:n), q0_sp(1:n));
  r_q1 = corr(q1_id(1:n), q1_sp(1:n));
  text(0.05, 0.95, sprintf('r=%.3f', min(r_q0, r_q1)), ...
    'Units', 'normalized', 'FontSize', 8, 'VerticalAlignment', 'top');

  xlabel('Identity IC'); ylabel('Spine IC');
  title(sprintf('Seed %d', s));

  fprintf('Seed %d: r_q0=%.4f  r_q1=%.4f\n', s, r_q0, r_q1);
end

sgtitle('IC Sequence Check (blue=q0, red=q1 — should be on diagonal)');
