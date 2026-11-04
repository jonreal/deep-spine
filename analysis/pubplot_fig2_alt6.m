% Cross-embodiment comparison: motif sweeps + example final-W / motif spectra.
%
% Left : reflex / synergy / antagonist motif strength over the (p_b, lambda_max)
%        sweep, p_b on x (log), lambda_max on y, one row per embodiment.
% Right: final weight matrix and motif spectrum at three operating points that
%        are COMMON to every embodiment (marked on every sweep):
%          canonical     (p_b = 0.0046, lambda_max = 0.2)   pentagram
%          higher babble (p_b = 0.0129, lambda_max = 0.5)   triangle
%          malformed     (p_b = 0.0028, lambda_max = 0.9)   diamond
%        Each body's run at a point is found from its summary table by matching
%        (lambda_max, p_b), so no run directories are hard-coded.
%
% A body whose summary table is missing ([] path) keeps its row, labels and
% axis furniture with no data drawn, so the layout never changes shape.

root = '../experiments/hebbian/logs/hebbian-sweep/';

% Hebbian gain slice to plot. Every sweep must contain this value.
heb_gain = 5;

% MR0 bodies only; the MR1 rows (mass ratio has little effect) go to the SI.
tags = {'SR0-MR0','SR1-MR0','SR2-MR0'};
paths_all = cellfun(@(t) {[root, t, '/analysis/summary/summary_table.mat']}, ...
                    tags, 'UniformOutput', false);

% operating points shown in the summary blocks (and marked on the sweeps)
pts = struct( ...
  'name',   {'Canonical', 'Higher babble', 'Malformed'}, ...
  'pb',     {0.0046,      0.0129,          0.0028}, ...
  'lam',    {0.2,         0.5,             0.9}, ...
  'marker', {'p',         '^',             'd'}, ...
  'msize',  {14,          9,               9});

% One font size for every label, tick and title: the figure is ~1450 px wide
% and prints at ~45% of that at full text width, so ~14 px -> ~6-7 pt.
fontsize = 14;
npts = numel(pts);

% Row label: strength-ratio name only; the ratios (and MR0 = 2:1, common to
% every row) are given in the caption.
sr_ratio = {'1:0.5:0.25','1:0.25:0.5','0.5:0.5:0.75'};
labels = cell(1, numel(tags));
for ii = 1:numel(tags)
  si = str2double(tags{ii}(3)) + 1;     % 'SR#-MR#' -> SR index
  labels{ii} = sprintf('SR_%d', si-1);
end

has_data = cellfun(@(p) exist(p{1}, 'file') == 2, paths_all);
if ~any(has_data)
  error('pubplot_fig2_alt5:noRows', 'No summary tables found under %s.', root);
end
fprintf('Rows with data: %s of 1..%d\n', ...
        strjoin(string(find(has_data)), ','), numel(has_data));

n = numel(tags);
tab = cell(n,1);
for i = 1:n
  if ~has_data(i), continue; end
  S = load(paths_all{i}{1}, 'summary_table');
  tab{i} = S.summary_table;
end

% =========================================================================
% Layout
% =========================================================================
nrows = n;
ncols = 3;

tab_ok = tab(has_data);
alllam = cell2mat(cellfun(@(T) T.opt_max_mus_str(:),     tab_ok, 'UniformOutput', false));
allpb  = cell2mat(cellfun(@(T) T.sup_kwa_bab_max_pro(:), tab_ok, 'UniformOutput', false));
% plot_heatmap log-scales y only, so p_b is passed as log10(p_b) on a linear
% x-axis (interpolation in log space) with 10^k tick labels.
xl = log10([min(allpb) max(allpb)]);
yl = [min(alllam) max(alllam)];
zl = [0 1];
fprintf('Axis limits from data: p_b %.4g..%.4g, lambda_max %.4g..%.4g\n', ...
        10^xl(1), 10^xl(2), yl(1), yl(2));

% Layout in PIXELS; the figure width is computed from the content so there is
% no slack. [pad | labels | 3 heatmaps | gap_div | 3 x (W | bars) | pad]
figH      = 470;      % row pitch matches the six-row version
ypad_px   = 46;       % top/bottom margin (two-line bar labels at the bottom)
pad_px    = 10;       % left/right figure margin
lab_px    = 60;       % left column: bottom-left y ticks + lambda_max label
xsp_px    = 14;       % gap between heatmaps
gapdiv_px = 58;       % gap between heatmap grid and first summary (colorbars)
gapWb_px  = 12;       % gap: W -> its motif bar
bar_px    = 125;      % motif-spectrum bar width
gapsum_px = 34;       % gap between summary blocks (room for right-side bar ticks)
lab_dx    = -50;       % SR label nudge: + right / - left (px)
lab_dy    = -15;       % SR label nudge: + up / - down (px)
sq_frac   = 0.82;     % heatmap square side / row pitch: leaves a vertical gap
                      % between rows (axis square would otherwise stack them)

ysp  = 0.001;
hm_px = sq_frac*(figH - 2*ypad_px)/nrows;
ytick_px = 16;        % right-side bar tick labels past the last block
figW = round(2*pad_px + ytick_px + lab_px + ncols*hm_px + (ncols-1)*xsp_px + gapdiv_px + ...
             npts*(hm_px + gapWb_px + bar_px) + (npts-1)*gapsum_px);
fig = figure('Position', [100 28 figW figH]);

% normalized equivalents used below
xpad    = pad_px/figW;
wlabs   = lab_px/figW;
xsp     = xsp_px/figW;
gap_div = gapdiv_px/figW;
ypad    = ypad_px/figH;
h = (1 - 2*ypad - (nrows-1)*ysp) / nrows;
w = hm_px/figW;
x_hm0      = xpad + wlabs;
x_hm_right = x_hm0 + ncols*w + (ncols-1)*xsp;
x_sum0     = x_hm_right + gap_div;

% weight-matrix colormap + alpha (match pubplot_fig1)
hmalpha = 0.5;
nW   = 2048;
blue = [linspace(0,1,nW/2)', linspace(0,1,nW/2)', ones(nW/2,1)];
red  = [ones(nW/2,1), linspace(1,0,nW/2)', linspace(1,0,nW/2)'];
cmapW = [blue; red];

cmapW = [red];



%cmapSweep = magma(1024);
cmapSweep = turbo(1024);
%cmapSweep = jet(1024);
%cmapSweep = viridis(1024);


% motif group colors + shading (match pubplot_fig1)
cvR = 0.6;  cvS = 0.0;  cvA = -0.9;
cmapP = parula;  nP = size(cmapP,1);
v2i = @(v) round((v - (-1)) / (1 - (-1)) * (nP - 1)) + 1;
cR = cmapP(v2i(cvR),:);  cS = cmapP(v2i(cvS),:);  cA = cmapP(v2i(cvA),:);
shade = linspace(0.5,1,3);
Cbar = [shade(1).*cR; shade(2).*cR; shade(3).*cR; ...
        shade(1).*cS; shade(2).*cS; shade(3).*cS; ...
        shade(1).*cA; shade(2).*cA; shade(3).*cA];
BW = 0.65;

% right-region geometry: one summary block per operating point
s_px = min(w*figW, h*figH);          % heatmap square side (px)
hbar = min(s_px/figH, 0.86*h);       % bar height ~ heatmap square
ybar = (h - hbar)/2;

wW_sum  = w;          % final-W width = sweep-heatmap width
gap_Wb  = gapWb_px/figW;
gap_sum = gapsum_px/figW;
wB_sum  = bar_px/figW;
stride  = wW_sum + gap_Wb + wB_sum + gap_sum;
xsW = @(k) x_sum0 + (k-1)*stride;
xsB = @(k) xsW(k) + wW_sum + gap_Wb;

zfields = {'r_mean','s_mean','a_mean'};
ctitles = {'Reflex','Synergy','Antagonist'};

sty = struct('cmapW',cmapW, 'hmalpha',hmalpha, 'Cbar',Cbar, 'BW',BW, ...
             'cGroup',[cR; cS; cA]);

ys = 1 - ypad - h;
for i = 1:n

  rowHasData = has_data(i);
  if rowHasData
    iq = find(tab{i}.('heb_kwa_gai') == heb_gain);
    if isempty(iq)
      error('pubplot_fig2_alt5:noGain', ...
        'Row %d (%s) has no entries at heb_kwa_gai = %g. Available: %s.', ...
        i, tags{i}, heb_gain, mat2str(unique(tab{i}.('heb_kwa_gai'))'));
    end
    x = log10(tab{i}.sup_kwa_bab_max_pro(iq));
    y = tab{i}.opt_max_mus_str(iq);
  end

  % ---- sweep heatmaps, with every operating point marked ----
  xs = x_hm0;
  for c = 1:ncols
    pos = [xs, ys, w, h];
    ax = axes('Position', pos);
    tl = (i==n) && (c==1);   % only the bottom-left sweep gets tick labels
    hmArgs = {'xl',xl, 'yl',yl, 'zl',zl, 'scale','linear', ...
              'XTick',[-3 -2 -1], 'YTick',linspace(0.1,1,4), ...
              'XTickLabel',{'10^{-3}','10^{-2}','10^{-1}'}, ...
              'YTickLabel',{'0.1','0.4','0.7','1'}, ...
              'TickDir','out', ...
              'XTickLabels',tl, 'YTickLabels',tl};
    if rowHasData
      plot_heatmap(ax, x, y, tab{i}.(zfields{c})(iq), hmArgs{:}, cmap=cmapSweep);
    else
      blank_heatmap(ax, hmArgs{:});
    end
    hold(ax,'on');
    for k = 1:npts
      plot3(ax, log10(pts(k).pb), pts(k).lam, zl(2), pts(k).marker, ...
        'MarkerFaceColor','w', 'MarkerEdgeColor','k', ...
        'MarkerSize',pts(k).msize, 'LineWidth',1);
    end

    if tl
      ax.XTickLabelRotation = 0;   % pin: MATLAB auto-rotates at large fonts
      xlabel(ax, 'p_b');
      ylabel(ax, '\lambda_{max}');
      ax.TickDir = 'out';
    end
    if i==1
      ann = annotation('textbox','Position',pos+[0,0.07,0,0]);   % above the SR label
      ann.String = ctitles{c};
      ann.EdgeColor = 'none';
      ann.HorizontalAlignment = 'center';
      ann.VerticalAlignment = 'top';
    end
    if c==1
      % row label just above the top-left corner of the Reflex map (the
      % square sits centred in its row, axis square), in the gap between rows
      y_top = ys + (h + w*figW/figH)/2;
      ann = annotation('textbox','Position', ...
        [xs + lab_dx/figW, y_top + lab_dy/figH, w, 0.05]);
      ann.String = labels{i};
      ann.EdgeColor = 'none';
      ann.Margin = 0;
      ann.HorizontalAlignment = 'left';
      ann.VerticalAlignment = 'bottom';
    end
    xs = xs + xsp + w;
  end

  % ---- one summary block per operating point ----
  if i==1
    for k = 1:npts
      block_header(xsW(k), xsB(k) + wB_sum, ys + h + 0.005, pts(k));
    end
  end
  for k = 1:npts
    if ~rowHasData, continue; end
    exDir = find_example(tab{i}, pts(k), heb_gain, [root, tags{i}]);
    if isempty(exDir), continue; end
    [finalW, D] = load_example(exDir);
    geo = struct('xsW',xsW(k), 'xsB',xsB(k), 'wW',wW_sum, 'wBar',wB_sum, ...
                 'h',h, 'ys',ys, 'ybar',ybar, 'hbar',hbar);
    draw_summary(finalW, D, geo, sty, i==n);
  end

  ys = ys - ysp - h;
end

% grey dividers: sweep grid | block 1 | block 2 | block 3
xdiv = (x_hm_right + x_sum0) / 2;
annotation('line', [xdiv xdiv], [ypad/2, 1-ypad/2], 'Color', [0.75 0.75 0.75]);
for k = 1:npts-1
  xd = xsB(k) + wB_sum + gap_sum/2;
  annotation('line', [xd xd], [ypad/2, 1-ypad/2], 'Color', [0.75 0.75 0.75]);
end

% colorscales in the bottom row, flanking the first divider
ys_bottom = 1 - ypad - h - (n-1)*(ysp+h);
cbw  = 7/figW;
cgap = 17/figW;
ycb  = ys_bottom + ybar;

axc = axes('Position', [xdiv - cgap - cbw, ycb, cbw, hbar]);   % sweeps (left)
ps = imagesc([0 1],[0 1], flipud(linspace(0,1,256)'));
ps.AlphaData = 0.8;
colormap(axc, cmapSweep);  caxis(axc,[0 1]);
ylim(axc,[0 1]);  axc.XAxis.Visible = 'off';
axc.YAxisLocation = 'right';
yticks(axc,[0 1]);  yticklabels(axc,{'1','0'});

axcw = axes('Position', [xdiv + cgap, ycb, cbw, hbar]);        % weights (right)
pw = imagesc([0 1],[0 1], flipud(linspace(0,1,256)'));
pw.AlphaData = hmalpha;
colormap(axcw, cmapW);  caxis(axcw,[0 1]);
ylim(axcw,[0 1]);  axcw.XAxis.Visible = 'off';
yticks(axcw,[0 1]);  yticklabels(axcw,{'1','0'});

% uniform font size across axes, tick labels, titles and annotations
set(findall(fig, '-property', 'FontSize'), 'FontSize', fontsize);
% S/E/B bar tick labels one size smaller (the r/s/a group labels are text)
for axb = findall(fig, 'Tag', 'motifBarBottom')'
  axb.XAxis.FontSize = fontsize - 1;
end

% =========================================================================
% Local helpers
% =========================================================================
function blank_heatmap(ax, opt)
  % The axis furniture of PLOT_HEATMAP with no surface drawn: used for rows
  % whose dataset is not available, so the grid keeps its exact geometry.
  arguments
    ax
    opt.xl = [0.1,1];
    opt.yl = [0.0003 0.0100];
    opt.zl = [0 1];
    opt.scale = 'log';
    opt.XTickLabels = false
    opt.YTickLabels = false
    opt.XTick = []
    opt.YTick = []
    opt.XTickLabel = {}
    opt.YTickLabel = {}
    opt.TickDir = 'in'
  end
  ax.YAxis.Scale = opt.scale;
  xlim(ax, opt.xl);  ylim(ax, opt.yl);  zlim(ax, opt.zl);
  view(ax, 0, 90);
  ax.Color = 'none';
  ax.TickDir = opt.TickDir;
  if ~isempty(opt.XTick), xticks(ax, opt.XTick); end
  if ~isempty(opt.YTick), yticks(ax, opt.YTick); end
  if opt.XTickLabels && ~isempty(opt.XTickLabel)
    xticklabels(ax, opt.XTickLabel);
    ax.XAxis.TickLabelInterpreter = 'tex';
  elseif ~opt.XTickLabels
    ax.XAxis.TickLabels = {};
  end
  if opt.YTickLabels && ~isempty(opt.YTickLabel)
    yticklabels(ax, opt.YTickLabel);
    ax.YAxis.TickLabelInterpreter = 'tex';
  elseif ~opt.YTickLabels
    ax.YAxis.TickLabels = {};
  end
  axis(ax, 'square');
end

function exDir = find_example(T, pt, heb_gain, bodyDir)
  % Run directory for this body at operating point pt (matched on the swept
  % lambda_max and p_b at the plotted Hebbian gain). Empty if absent.
  k = find(abs(T.opt_max_mus_str - pt.lam) < 1e-6 & ...
           abs(T.sup_kwa_bab_max_pro - pt.pb) < 1e-6 & ...
           T.('heb_kwa_gai') == heb_gain, 1);
  if isempty(k)
    warning('pubplot_fig2_alt5:noPoint', 'No run at p_b=%g, lambda_max=%g in %s', ...
            pt.pb, pt.lam, bodyDir);
    exDir = '';
    return
  end
  exDir = fullfile(bodyDir, char(T.('sim-ID'){k}));
  fprintf('%-11s %s\n', pt.name, exDir);
end

function [finalW, D] = load_example(exampleDir)
  % Final weight matrix and the (last-200-step mean) motif spectrum.
  [~,~,~,~,We] = extract_data([exampleDir,'/data.txt']);
  motif  = ds_Ia_motif(We);
  finalW = We{end};
  w200 = max(1, size(motif.rp,1)-199) : size(motif.rp,1);
  D = [mean(motif.rp(w200,:),1), ...
       mean(motif.sp(w200,:),1), ...
       mean(motif.ap(w200,:),1)]';
end

function block_header(x0, x1, y, pt)
  % Operating-point coordinates centred over a summary block. The marker
  % symbol is added in LaTeX (overpic).
  ann = annotation('textbox', [x0, y, x1 - x0, 0.02]);
  ann.String = sprintf('p_b = %g, \\lambda_{max} = %g', pt.pb, pt.lam);
  ann.EdgeColor = 'none';
  ann.HorizontalAlignment = 'center';
  ann.VerticalAlignment = 'middle';
  ann.FontSize = 10;
end

function draw_summary(finalW, D, geo, sty, isBot)
  % One summary block: final weight matrix + motif-spectrum bar.
  axW = axes('Position', [geo.xsW, geo.ys, geo.wW, geo.h]);
  p = imagesc(finalW);
  p.AlphaData = sty.hmalpha;
  axis square; axis off;
  colormap(axW, sty.cmapW);
  caxis(axW, [0 1]);

  axB = axes('Position', [geo.xsB, geo.ys+geo.ybar, geo.wBar, geo.hbar]);
  b = bar(D, 'LineWidth',0.25, 'FaceColor','flat');
  b.BarWidth  = sty.BW;
  b.CData     = sty.Cbar;
  b.EdgeColor = 'none';
  ylim([0 1]); xlim([0.5 9.5]);
  axB.Color = 'none';  grid on;
  axB.YAxisLocation = 'right';
  yticks([0 0.5 1]);
  xticks(1:9);
  if isBot
    % two-line labels: muscle group under each bar, motif under each triplet
    yticklabels({'0','','1'});
    xticklabels(repmat({'S','E','B'}, 1, 3));
    axB.Tag = 'motifBarBottom';
    xtickangle(axB, 0);
    grp = {'r','s','a'};
    for g = 1:3
      text(axB, 3*g-1, -0.2, grp{g}, 'Color', sty.cGroup(g,:), ...
           'HorizontalAlignment','center', 'VerticalAlignment','top', ...
           'FontWeight','bold');
    end
  else
    yticklabels({});
    xticklabels({});
  end
end
