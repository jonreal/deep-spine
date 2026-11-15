% PUBPLOTS_SI_3  SI figures: Hebbian-gain (kappa) progression, SR0-MR0 body.
%
% Script. Loads the three SR0-MR0 sweep summary tables (kappa = 1-4, 5, 10),
% merges them, and draws
%   figure 1 : mean reflex / synergy / antagonist vs p_b at lambda_max = lamSlice,
%              one line per kappa                         (-> si-kappa-lines)
%   figure 2 : ds_hm_motif heatmaps (R, S, A), one row per kappa (-> si-kappa-heatmaps)
%
% Figure handles are left in the workspace as figLines / figHeat for export.

root = '../experiments/hebbian/logs/hebbian-sweep/';
srcs = {'SR0-MR0-gain', 'SR0-MR0', 'SR0-MR0-gain-10'};
lamSlice = 0.2;

fontsize = 12;
LW       = 1.5;

% ---------------------------------------------------
% Load + merge (keep only the columns common to all tables)
% ---------------------------------------------------
tabs = cell(size(srcs));
for k = 1:numel(srcs)
  L = load(fullfile(root, srcs{k}, 'analysis', 'summary', 'summary_table.mat'), 'summary_table');
  tabs{k} = L.summary_table;
end
names = tabs{1}.Properties.VariableNames;
for k = 2:numel(tabs), names = intersect(names, tabs{k}.Properties.VariableNames, 'stable'); end
Tk = vertcat(tabs{1}(:,names), tabs{2}(:,names), tabs{3}(:,names));

kappas = unique(Tk.heb_kwa_gai)';
fprintf('kappa values: %s\n', mat2str(kappas));

% ---------------------------------------------------
% Figure 1: lambda_max slice, one line per kappa
% ---------------------------------------------------
% grey ramp light -> dark with kappa; the gain used in the paper (5) in black
cK = repmat(linspace(0.8, 0.35, numel(kappas))', 1, 3);
cK(kappas == 5, :) = 0;

figLines = figure('Position', [100 100 1100 330], 'Color', 'w');
tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
fld = {'r_mean', 's_mean', 'a_mean'};
ttl = {'Reflex (R)', 'Synergy (S)', 'Antagonist (A)'};
for c = 1:3
  a = nexttile(tl);  hold(a, 'on');
  for k = 1:numel(kappas)
    sel = Tk.heb_kwa_gai == kappas(k) & abs(Tk.opt_max_mus_str - lamSlice) < 1e-6;
    [pb, o] = sort(Tk.sup_kwa_bab_max_pro(sel));
    y = Tk.(fld{c})(sel);  y = y(o);
    lw = LW;  if kappas(k) == 5, lw = 2.2; end
    plot(a, pb, y, '-o', 'Color', cK(k,:), 'MarkerFaceColor', cK(k,:), ...
         'MarkerSize', 4, 'LineWidth', lw);
  end
  a.XScale = 'log';  xlim(a, [1e-3 1e-1]);  ylim(a, [0 1]);
  xticks(a, [1e-3 1e-2 1e-1]);
  grid(a, 'on');  box(a, 'on');  a.FontSize = fontsize;
  title(a, ttl{c}, 'FontWeight', 'normal');
  xlabel(a, 'p_b');
  if c == 1, ylabel(a, 'Motif strength'); else, a.YTickLabel = []; end
end
lgd = legend(a, compose('\\kappa = %g', kappas), 'Box', 'off', 'FontSize', fontsize-1);
lgd.Layout.Tile = 'east';
title(tl, sprintf('SR_0, MR_0, \\lambda_{max} = %g', lamSlice), 'FontSize', fontsize);

% ---------------------------------------------------
% Figure 2: heatmaps per kappa (existing ds_hm_motif)
% ---------------------------------------------------
ds_hm_motif(Tk, selectivity=false, cmap=turbo(1024));   % turbo, as Fig 5
figHeat = gcf;
