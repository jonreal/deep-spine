% PUBPLOT_FIG4  Fig 4 in one figure: F matrix, (optional) -log10 p matrix,
% and F by connection class. Rendered together so all panels share font sizes.
%
% Script -- expects in the workspace:
%   W  high-rate weights (from ds_load_trial), used to classify connections
%   R  output of gc_batch on the same trial
% e.g.
%   root = '../experiments/hebbian/logs/example-3/1790298957';
%   [S,W,P] = ds_load_trial(root);
%   dW = ds_load_dw_rule(root);
%   R = gc_batch(S,W,loFrac=0,hiFrac=0.9,lag=0.25,grid=false,plot=false,dWrule=dW);
%   pubplot_fig4
%
% Panel letters are added in LaTeX. Untested connections (antagonists, NaN)
% are blank. Strip-plot classes and colours follow pubplot_gc_stats.

if ~exist('W','var') || ~exist('R','var')
  error('pubplot_fig4:noData', ...
    'Need W (ds_load_trial) and R (gc_batch) in the workspace.');
end

% ---------------------------------------------------
% Parameters
% ---------------------------------------------------
showP     = false;        % include the -log10 p matrix panel
showTitle = false;        % titles above the matrix panels (off: panels are
                          % identified by LaTeX letters and the caption)
pcap      = 300;          % -log10 p display cap (p underflows to 0 near 1e-308)
pSig      = 0.01;         % significance threshold for filled dots
tTrans    = 0.05;         % fraction of run treated as the initial transient
dForm     = 0.05;         % min rise above initial value to count as forming
grow_frac = 0.8;          % final >= grow_frac*max_post -> growing
jitterW   = 0.18;         % horizontal jitter half-width
MS        = 5;
fontsize  = 8;
fwidth    = 504;          % pt, full text width
fheight   = 165;          % pt
cmapM     = linspace(0.92, 0.05, 256)' * [1 1 1];   % as gc_batch

labels = {'SF','SE','EF','EE','BF','BE'};

% ---------------------------------------------------
% Colors (as Fig 2: parula anchors, shaded by motoneuron pool)
% ---------------------------------------------------
cmap1 = parula;
nColors = size(cmap1,1);
val2idx = @(v) round((v - (-1)) / (1 - (-1)) * (nColors - 1)) + 1;
cR = cmap1(val2idx(0.6),:);
cS = cmap1(val2idx(0.0),:);
cA = cmap1(val2idx(-0.9),:);
shade = linspace(0.5,1,3);

% ---------------------------------------------------
% Classify (as pubplot_gc_stats)
% ---------------------------------------------------
[nMN, nIa, T] = gc_wsize(W);
iT0 = max(1, round(tTrans*T));
class_names = {'Reflex','growing','rise-fall','flat','Antagonist'};
class_col   = {cR, cS, cS, cS, cA};
cls = zeros(nMN, nIa);
for i = 1:nMN
  for j = 1:nIa
    if i == j
      cls(i,j) = 1;
    elseif mod(i+j,2) == 1
      cls(i,j) = 5;
    else
      wij = double(gc_wtrace(W, i, j));
      mx_post = max(wij(iT0:end));
      if mx_post < wij(1) + dForm
        cls(i,j) = 4;
      elseif wij(end) >= grow_frac*mx_post
        cls(i,j) = 2;
      else
        cls(i,j) = 3;
      end
    end
  end
end
present = arrayfun(@(c) any(cls(:)==c & ~isnan(R.F_Ia(:))), 1:numel(class_names));
cidx    = find(present);

% ---------------------------------------------------
% Layout (points)
% ---------------------------------------------------
fig = figure('Units','points','Position',[100 100 fwidth fheight],'Color','w');
% Each matrix is followed by its colorbar (~30 pt incl. ticks); each axes
% needs ~40 pt on its left for tick labels and the y label.
mBot = 30; mTop = 14; hAx = fheight - mBot - mTop;
if showP
  hM = 100;  xM = [40, 40 + hM + 30 + 40];
else
  hM = hAx;  xM = 40;
end
xS = xM(end) + hM + 30 + 42;  wStrip = fwidth - xS - 4;
yM = mBot + (hAx - hM)/2;

% ---------------------------------------------------
% Matrix panels
% ---------------------------------------------------
P = -log10(max(R.p_Ia, realmin));  P(isnan(R.p_Ia)) = NaN;
mats = {R.F_Ia, 'F', [0 max(R.F_Ia(:),[],'omitnan')]; ...
        P, '-log_{10} p', [0 pcap]};
for m = 1:numel(xM)
  ax = axes(fig, 'Units','points', 'Position', [xM(m) yM hM hM]);
  M = mats{m,1};  bad = isnan(M);  M(bad) = 0;
  him = imagesc(ax, M);  set(him, 'AlphaData', ~bad);
  ax.Color = 'w';  axis(ax, 'image');
  colormap(ax, cmapM);  clim(ax, mats{m,3});
  cb = colorbar(ax);  cb.FontSize = fontsize;
  cb.Units = 'points';  cb.Position(1) = xM(m) + hM + 4;  cb.Position(3) = 6;
  xticks(ax, 1:nIa);  yticks(ax, 1:nMN);
  xticklabels(ax, labels);  yticklabels(ax, labels);
  xlabel(ax, 'Ia afferent');  ylabel(ax, 'Motoneuron');
  if showTitle, title(ax, mats{m,2}, 'FontWeight', 'normal'); end
  ax.FontSize = fontsize;  ax.TickLength = [0 0];
end

% ---------------------------------------------------
% Strip plot: F by class
% ---------------------------------------------------
ax = axes(fig, 'Units','points', 'Position', [xS mBot wStrip hAx]);
hold(ax, 'on');
rng(0);                                   % reproducible jitter
for x = 1:numel(cidx)
  c = cidx(x);
  [I,J] = find(cls == c & ~isnan(R.F_Ia));
  for k = 1:numel(I)
    col = shade(ceil(I(k)/2)) * class_col{c};
    xj  = x + (rand*2 - 1)*jitterW;
    if R.p_Ia(I(k),J(k)) < pSig
      plot(ax, xj, R.F_Ia(I(k),J(k)), 'o', 'MarkerSize', MS, ...
           'MarkerFaceColor', col, 'MarkerEdgeColor', col);
    else
      plot(ax, xj, R.F_Ia(I(k),J(k)), 'o', 'MarkerSize', MS, ...
           'MarkerFaceColor', 'w', 'MarkerEdgeColor', col, 'LineWidth', 1);
    end
    text(ax, xj + 0.08, R.F_Ia(I(k),J(k)), ...
         sprintf('%s\\leftarrow%s', labels{I(k)}, labels{J(k)}), ...
         'FontSize', fontsize-2, 'Color', [0.4 0.4 0.4]);
  end
end
yline(ax, 1, '--', 'Color', [0.5 0.5 0.5]);
xlim(ax, [0.5 numel(cidx) + 0.6]);
ylim(ax, [0 max(R.F_Ia(:), [], 'omitnan')*1.1]);
xticks(ax, 1:numel(cidx));
xticklabels(ax, class_names(cidx));
ylabel(ax, 'F (kinematics | Ia + history)');
ax.FontSize = fontsize;
box(ax, 'on');
hf = plot(ax, NaN, NaN, 'o', 'MarkerFaceColor', [0.4 0.4 0.4], ...
          'MarkerEdgeColor', [0.4 0.4 0.4], 'MarkerSize', MS);
ho = plot(ax, NaN, NaN, 'o', 'MarkerFaceColor', 'w', ...
          'MarkerEdgeColor', [0.4 0.4 0.4], 'MarkerSize', MS);
legend(ax, [hf ho], {sprintf('p < %g', pSig), 'n.s.'}, ...
       'Location', 'northeast', 'Box', 'off', 'FontSize', fontsize-1);
