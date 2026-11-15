function figS3_timehistory(data_root, out_dir, trial)
%FIGS3_TIMEHISTORY  SI figure: multi-scale time history of the canonical run.
%
%   figS3_timehistory()                          canonical run from <repo>/logs
%   figS3_timehistory(data_root, out_dir, trial) trial from LOAD_CANONICAL (reused)
% Writes <out_dir>/si-timehistory.pdf.
%
% Symmetric zoom cascade, full run in the centre:
%   [early 1 s | early 10 s | full 50 h | late 10 s | late 1 s]
%
% Only the muscles of three example connections are shown, all sharing the
% presynaptic afferent Ia_SF:  SF<-SF (reflex), BF<-SF (synergy),
% SE<-SF (antagonist). Postsynaptic signals are colored by connection class.
%
% Rows: babble drive, motoneurons, Ia_SF, joint angles, joint velocities,
% learning state. The learning row changes quantity with timescale:
%   full run : weights w_ij, with the strength ramp lambda(t) (right axis)
%   10 s     : eligibility l_ij
%   1 s      : instantaneous Hebbian term x_Ia,j * w_ij * xhat_MN,i, with xhat the
%              postsynaptic term kappa*HP(x_MN,i) - xbar_MN,i (Fig 1 notation)
% which is what l_ij low-pass integrates. l_ij and the Hebbian term were not
% logged; they are recomputed exactly from the logged x_MN, x_Ia and W with
% the learner's own filters (hebbian_kwargs in the run's config.json).
%
% Windows are chosen by rule, not by eye:
%   early 10 s : centred on the steepest growth of the reflex (SF<-SF)
%   late 10 s  : centred on the steepest growth of the synergy (BF<-SF),
%                searched within lateRange of the run (hold phase)
%   1 s        : centred on the largest single weight increase of that
%                connection inside its 10 s window, excluding the outer
%                win_sec at each end so the 1 s band sits inside the panel

arguments
  data_root (1,:) char = ''
  out_dir (1,:) char = ''
  trial struct = struct()
end
[data_root, out_dir] = pubfigs_setup(data_root, out_dir);
if isempty(fieldnames(trial)), trial = load_canonical(data_root, dW=false); end
S = trial.S;  W = trial.W;  P = trial.P;

% ---------------------------------------------------
% Parameters
% ---------------------------------------------------
fs        = P.fs;
N         = size(S.act_MN, 1);
T_s       = N/fs;
win_mid   = 10;          % middle zoom window (s)
win_sec   = 1;           % inner zoom window (s)
smoothS   = 60;          % smoothing of the 1 Hz weight rate for window picks (s)
lateRange = [0.6 0.8];   % fraction of the run searched for the late window
epSec     = 90;          % episode length for full-run statistics (s)
epSmooth  = 11;          % moving mean over episodes, full-run column (odd: centred; matches Fig 3)
driveOn   = 1e-3;        % drive above this counts as "on"

fontsize = 11;
LW       = 1.2;
fwidth   = 1600;
fheight  = 950;

% example connections: MN i <- Ia j (all share Ia_SF)
conn = struct( ...
  'name',  {'SF\leftarrowSF', 'BF\leftarrowSF', 'SE\leftarrowSF'}, ...
  'mus',   {'SF', 'BF', 'SE'}, ...
  'i',     {1, 5, 2}, ...
  'j',     {1, 1, 1}, ...
  'cls',   {'reflex', 'synergy', 'antagonist'});
jPre   = 1;              % shared presynaptic afferent (Ia_SF)
iEarly = 1;              % connection that picks the early windows (reflex)
iLate  = 2;              % connection that picks the late windows (synergy)

% ---------------------------------------------------
% Colors (Fig 2 motif anchors for the three connections / targets)
% ---------------------------------------------------
cmap1 = parula;
nColors = size(cmap1,1);
val2idx = @(v) round((v - (-1)) / (1 - (-1)) * (nColors - 1)) + 1;
cConn  = [cmap1(val2idx(0.6),:); cmap1(val2idx(0.0),:); cmap1(val2idx(-0.9),:)];
cPre   = [0 0 0];
cJoint = [0.50 0.50 0.50; 0.78 0.78 0.78];   % shoulder, elbow (kept lighter than Ia_SF, black)
cLam   = [0.5 0.5 0.5];

% ---------------------------------------------------
% Learning signals (recomputed with the learner's filters)
% ---------------------------------------------------
hk = P.config.hebbian_kwargs;
if iscell(hk), hk = hk{1}; else, hk = hk(1); end
kappa  = hk.gain;
Kmean  = hk.Kmean;
Klearn = hk.Klearn;
[bh, ah] = butter(hk.synaptic_filter.order, hk.synaptic_filter.cutoff, ...
                  hk.synaptic_filter.type);

nc   = numel(conn);
wtr  = zeros(N, nc, 'single');     % weight
heb  = zeros(N, nc, 'single');     % instantaneous Hebbian term
ell  = zeros(N, nc, 'single');     % eligibility
pre  = double(S.act_Ia(:, jPre));
for c = 1:nc
  post = double(S.act_MN(:, conn(c).i));
  wij  = double(ds_w_trace(W, conn(c).i, conn(c).j));  wij = wij(:);
  h    = pre .* wij .* ...
         (kappa*filter(bh, ah, post) - filter([0 Kmean], [1 Kmean-1], post));
  heb(:,c) = single(h);
  ell(:,c) = single(filter([0 Klearn], [1 Klearn-1], h));
  wtr(:,c) = single(wij);
end
clear post wij h

% ---------------------------------------------------
% Window selection (by rule)
% ---------------------------------------------------
t1 = (0:fs:N-1)' + 1;                            % 1 Hz sample indices
n1 = numel(t1);

rate = movmean(diff(double(wtr(t1, iEarly))), smoothS);
[~, kE] = max(rate);
rate = movmean(diff(double(wtr(t1, iLate))), smoothS);
kr   = max(1, round(lateRange(1)*n1)) : min(numel(rate), round(lateRange(2)*n1));
[~, kL] = max(rate(kr));  kL = kr(kL);

midE = pick_window(t1(kE), win_mid, fs, N);
midL = pick_window(t1(kL), win_mid, fs, N);
[~, pE] = max(interior(gradient(double(wtr(midE, iEarly))), round(win_sec*fs)));
[~, pL] = max(interior(gradient(double(wtr(midL, iLate))),  round(win_sec*fs)));
secE = pick_window(midE(pE), win_sec, fs, N);
secL = pick_window(midL(pL), win_sec, fs, N);

fprintf('early 10 s: %.2f h   early 1 s: %.4f h\n', midE(1)/fs/3600, secE(1)/fs/3600);
fprintf('late  10 s: %.2f h   late  1 s: %.4f h\n', midL(1)/fs/3600, secL(1)/fs/3600);

% strength ramp: P.muscle_strength is sampled at 1 Hz (one value per second)
lam_h = (0:numel(P.muscle_strength)-1)'/3600;
lamAt = @(idx) P.muscle_strength(min(numel(P.muscle_strength), floor(idx(1)/fs) + 1));

% ---------------------------------------------------
% Full-run statistics per episode (smoothed)
% ---------------------------------------------------
epN  = round(epSec*fs);
nep  = floor(N/epN);
ep_h = ((1:nep) - 0.5)*epSec/3600;
% nep x k (reshape, not squeeze: squeeze leaves a single column as a row)
epStat = @(X, f) movmean(reshape(f(reshape(double(X(1:nep*epN,:)), epN, nep, [])), nep, []), epSmooth, 1);

iT = [conn.i];
full.drive = epStat(S.drive(:, iT) > driveOn, @(Y) mean(Y, 1));   % on-fraction
full.MN    = epStat(S.act_MN(:, iT),          @(Y) mean(Y, 1));
full.Ia    = epStat(S.act_Ia(:, jPre),        @(Y) mean(Y, 1));
full.q     = epStat(S.qpos,                   @(Y) sqrt(mean(Y.^2, 1)));
full.qd    = epStat(S.qvel,                   @(Y) sqrt(mean(Y.^2, 1)));
full.w     = double(wtr(t1, :));
full.w_h   = (t1 - 1)/fs/3600;

% ---------------------------------------------------
% Figure
% ---------------------------------------------------
cols = {secE, midE, [], midL, secL};            % [] = full run
kind = {'sec', 'mid', 'full', 'mid', 'sec'};
ctit = {sprintf('Early, 1 s (\\lambda = %.3g)',  lamAt(secE)), ...
        sprintf('Early, 10 s (\\lambda = %.3g)', lamAt(midE)), ...
        'Full run (50 h)', ...
        sprintf('Late, 10 s (\\lambda = %.3g)',  lamAt(midL)), ...
        sprintf('Late, 1 s (\\lambda = %.3g)',   lamAt(secL))};
rowLb = {'Drive $d$', '$x_{\mathrm{MN}}$', '$x_{\mathrm{Ia,SF}}$', ...
         '$q$ (rad)', '$\dot{q}$ (rad/s)', 'Learning'};
nrow = numel(rowLb);  ncol = numel(cols);

% shared y limits across matching zoom panels
zoomIdx = [secE; midE; midL; secL];
pad   = @(l) l + [-0.05 0.05]*diff(l);
qlim  = pad([min(S.qpos(zoomIdx,:),[],'all'), max(S.qpos(zoomIdx,:),[],'all')]);
qdlim = pad([min(S.qvel(zoomIdx,:),[],'all'), max(S.qvel(zoomIdx,:),[],'all')]);
elim  = 1.1*max(abs(ell([midE; midL],:)),[],'all') * [-1 1];
hlim  = 1.1*max(abs(heb([secE; secL],:)),[],'all') * [-1 1];

fig = figure('Position', [50 50 fwidth fheight], 'Color', 'w');
tl = tiledlayout(nrow, ncol, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = gobjects(nrow, ncol);

for c = 1:ncol
  idx = cols{c};
  switch kind{c}
    case 'full', tt = [];
    case 'mid',  tt = (idx - idx(1))/fs;              % s
    case 'sec',  tt = (idx - idx(1))/fs*1000;         % ms
  end

  for r = 1:nrow
    a = nexttile(tl, (r-1)*ncol + c);  hold(a, 'on');  ax(r,c) = a;
    isFull = strcmp(kind{c}, 'full');
    switch r
      case 1   % drive (full: on-fraction)
        for k = 1:nc
          if isFull, plot(a, ep_h, full.drive(:,k), 'Color', cConn(k,:), 'LineWidth', LW);
          else,      plot(a, tt, S.drive(idx, conn(k).i), 'Color', cConn(k,:), 'LineWidth', LW); end
        end
        if ~isFull, ylim(a, [0 1]); end
      case 2   % motoneurons
        for k = 1:nc
          if isFull, plot(a, ep_h, full.MN(:,k), 'Color', cConn(k,:), 'LineWidth', LW);
          else,      plot(a, tt, S.act_MN(idx, conn(k).i), 'Color', cConn(k,:), 'LineWidth', LW); end
        end
        if ~isFull, ylim(a, [0 1]); end
      case 3   % shared afferent
        if isFull, plot(a, ep_h, full.Ia, 'Color', cPre, 'LineWidth', LW);
        else,      plot(a, tt, S.act_Ia(idx, jPre), 'Color', cPre, 'LineWidth', LW); ylim(a, [0 1]); end
      case {4,5}   % joint angle / velocity (full: RMS)
        for k = 1:2
          if isFull
            y = full.q(:,k);  if r == 5, y = full.qd(:,k); end
            plot(a, ep_h, y, 'Color', cJoint(k,:), 'LineWidth', LW);
          else
            src = S.qpos;  if r == 5, src = S.qvel; end
            plot(a, tt, src(idx,k), 'Color', cJoint(k,:), 'LineWidth', LW);
          end
        end
        if ~isFull
          if r == 4, ylim(a, qlim); else, ylim(a, qdlim); end
        end
      case 6   % learning
        switch kind{c}
          case 'full'
            for k = 1:nc, plot(a, full.w_h, full.w(:,k), 'Color', cConn(k,:), 'LineWidth', LW); end
            ylim(a, [0 1]);
            yyaxis(a, 'right');
            plot(a, lam_h, P.muscle_strength, '--', 'Color', cLam, 'LineWidth', LW);
            ylim(a, [0 1.1*max(P.muscle_strength)]);
            a.YAxis(2).Color = cLam;
            ylabel(a, '$\lambda$', 'Interpreter', 'latex');
            yyaxis(a, 'left');
            qty = '$w_{ij}$';
          case 'mid'
            for k = 1:nc, plot(a, tt, ell(idx,k), 'Color', cConn(k,:), 'LineWidth', LW); end
            ylim(a, elim);  qty = '$\ell_{ij}$';
          case 'sec'
            for k = 1:nc, plot(a, tt, heb(idx,k), 'Color', cConn(k,:), 'LineWidth', LW); end
            ylim(a, hlim);
            % hat-x as in Fig 1: the whole postsynaptic term kappa*HP(x) - xbar
            qty = '$x_{\mathrm{Ia}}\,w\,\hat{x}_{\mathrm{MN}}$';
        end
        if ~isFull, yline(a, 0, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off'); end
        text(a, 0.02, 0.92, qty, 'Units', 'normalized', 'Interpreter', 'latex', ...
             'FontSize', fontsize-1, 'Color', [0.35 0.35 0.35]);
    end

    % axes furniture
    if isFull, xlim(a, [0 T_s/3600]); else, xlim(a, tt([1 end])); end
    a.FontSize = fontsize;  box(a, 'on');
    if r < nrow, a.XTickLabel = []; end
    if r == 1, title(a, ctit{c}, 'FontWeight', 'bold'); end
    if c == 1, ylabel(a, rowLb{r}, 'Interpreter', 'latex'); end
  end

  switch kind{c}
    case 'full', xlabel(ax(nrow,c), 'Time (h)');
    case 'mid',  xlabel(ax(nrow,c), 'Time (s)');
    case 'sec',  xlabel(ax(nrow,c), 'Time (ms)');
  end
end

% full-run column: what each statistic is
% placed just above the axes' top edge (outside the data) so it never
% covers a trace
note = @(a, s) text(a, 1, 1, s, 'Units', 'normalized', ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
     'FontSize', fontsize-2, 'Color', [0.4 0.4 0.4]);
% top row: below the axes, left -- the E/L labels occupy the space above it
text(ax(1,3), 0, 0, 'on-fraction', 'Units', 'normalized', ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
     'FontSize', fontsize-2, 'Color', [0.4 0.4 0.4]);
note(ax(2,3), 'episode mean');
note(ax(3,3), 'episode mean');
note(ax(4,3), 'episode RMS');
note(ax(5,3), 'episode RMS');

% zoom guides: vertical lines through the full-run column, bands on 10 s panels
xE = mean(midE)/fs/3600;  xL = mean(midL)/fs/3600;
for r = 1:nrow
  xline(ax(r,3), xE, '--', 'Color', [0.3 0.3 0.3], 'HandleVisibility', 'off');
  xline(ax(r,3), xL, '--', 'Color', [0.3 0.3 0.3], 'HandleVisibility', 'off');
  mark_band(ax(r,2), (secE(1)-midE(1))/fs, (secE(end)-midE(1))/fs);
  mark_band(ax(r,4), (secL(1)-midL(1))/fs, (secL(end)-midL(1))/fs);
end
text(ax(1,3), xE, ax(1,3).YLim(2), 'E', 'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
text(ax(1,3), xL, ax(1,3).YLim(2), 'L', 'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'bottom', 'FontWeight', 'bold');

% one legend on the layout's east edge
hl = gobjects(nc + 4, 1);
for k = 1:nc, hl(k) = plot(ax(1,1), NaN, NaN, 'Color', cConn(k,:), 'LineWidth', 1.5); end
hl(nc+1) = plot(ax(1,1), NaN, NaN, 'Color', cPre,        'LineWidth', 1.5);
hl(nc+2) = plot(ax(1,1), NaN, NaN, 'Color', cJoint(1,:), 'LineWidth', 1.5);
hl(nc+3) = plot(ax(1,1), NaN, NaN, 'Color', cJoint(2,:), 'LineWidth', 1.5);
hl(nc+4) = plot(ax(1,1), NaN, NaN, '--', 'Color', cLam, 'LineWidth', 1.5);
legLab = [arrayfun(@(k) sprintf('%s (%s)', conn(k).mus, conn(k).name), 1:nc, 'UniformOutput', false), ...
          {'Ia_{SF}', 'Shoulder', 'Elbow', '\lambda'}];
lgd = legend(ax(1,1), hl, legLab, 'Box', 'off', 'FontSize', fontsize-1);
lgd.Layout.Tile = 'east';

save_pdf(fig, fullfile(out_dir, 'si-timehistory.pdf'));
end

% =========================================================================
% Local helpers
% =========================================================================
function idx = pick_window(center, winSec, fs, N)
  % sample indices of a winSec-second window centred on sample CENTER,
  % shifted to stay inside [1, N]
  n  = round(winSec*fs);
  i0 = max(1, min(N - n + 1, center - floor(n/2)));
  idx = (i0 : i0 + n - 1)';
end

function g = interior(g, m)
  % mask the first and last m samples so an argmax lands inside the window
  g([1:m, end-m+1:end]) = -Inf;
end

function mark_band(a, x0, x1)
  % light band marking the inner zoom window
  yl = a.YLim;
  patch(a, [x0 x1 x1 x0], yl([1 1 2 2]), [0.2 0.2 0.2], ...
        'FaceAlpha', 0.10, 'EdgeColor', 'none', 'HandleVisibility', 'off');
  a.YLim = yl;
end
