function fig3_grid(data_root, out_dir, trial)
%FIG3_GRID  Fig 3: Ia->MN weight trajectories as a 6x6 grid, with the
% muscle-level motif time courses and limb kinematics as extra columns.
%
%   fig3_grid()                          canonical run from <repo>/logs
%   fig3_grid(data_root, out_dir, trial) trial from LOAD_CANONICAL (reused)
% Writes <out_dir>/grid.pdf.
%
% Columns 1-6: w_ij(t) for MN i (rows) <- Ia j (columns), each trace colored
%   by its motif class and muscle pool exactly as in the Fig 2 legend.
% Column 7:    pool motif coefficients (reflex, synergy, antagonist) for the
%   S, E, B pools, from DS_FIT_MOTIF on the decimated weights (the same
%   computation as Fig 2).
% Column 8:    RMS joint angle (excursion from the q = 0 rest posture) and
%   RMS joint velocity per window, shoulder and elbow.

arguments
  data_root (1,:) char = ''
  out_dir (1,:) char = ''
  trial struct = struct()
end
[data_root, out_dir] = pubfigs_setup(data_root, out_dir);
if isempty(fieldnames(trial)), trial = load_canonical(data_root, dW=false); end
S = trial.S;  W = trial.W;

% ---------------------------------------------------
% Parameters
% ---------------------------------------------------
fs        = 200;          % high-rate log (Hz)
plotN     = 2000;         % points drawn per trace
winSec    = 90;           % RMS window for kinematics (one episode)
smoothWin = 11;           % movmean over RMS windows (episodes)
nTick     = 5;            % x ticks (matches Fig 2 snapshot ticks)

LW       = 1;
fontsize = 8;
fwidth   = 790;
fheight  = 440;

labels   = {'SF','SE','EF','EE','BF','BE'};
pool_lab = {'S','E','B'};

% ---------------------------------------------------
% Colors (Fig 2: parula anchors, shaded by muscle pool S/E/B)
% ---------------------------------------------------
cmap1 = parula;
nColors = size(cmap1,1);
val2idx = @(v) round((v - (-1)) / (1 - (-1)) * (nColors - 1)) + 1;
cR = cmap1(val2idx(0.6),:);
cS = cmap1(val2idx(0.0),:);
cA = cmap1(val2idx(-0.9),:);
shade = linspace(0.5,1,3);
cKin  = [0.3 0.3 0.3; 0.65 0.65 0.65];   % shoulder, elbow

% ---------------------------------------------------
% Decimated weights -> motif regression
% ---------------------------------------------------
[nMN, nIa, T] = ds_w_size(W);
dp  = max(1, ceil(T/plotN));
ip  = 1:dp:T;
t_h = ip/fs/3600;

Wdec = zeros(numel(ip), nMN, nIa);
for i = 1:nMN
  for j = 1:nIa
    wij = ds_w_trace(W, i, j);
    Wdec(:,i,j) = double(wij(ip));
  end
end
We    = arrayfun(@(k) squeeze(Wdec(k,:,:)), 1:numel(ip), 'UniformOutput', false);
motif = ds_fit_motif(We);

% per-cell color from the pool masks (columns: r_S r_E r_B s_S ... a_B)
cls_col = {cR, cS, cA};
cellCol = zeros(nMN, nIa, 3);
for k = 1:9
  m = reshape(motif.maskp(:,k), nMN, nIa);
  c = shade(mod(k-1,3)+1) * cls_col{ceil(k/3)};
  for ch = 1:3
    cellCol(:,:,ch) = cellCol(:,:,ch) + m*c(ch);
  end
end

% ---------------------------------------------------
% Kinematics: RMS per window
% ---------------------------------------------------
win  = round(winSec*fs);
nwin = floor(size(S.qpos,1)/win);
qrms = zeros(nwin, size(S.qpos,2));
vrms = zeros(nwin, size(S.qvel,2));
for k = 1:size(S.qpos,2)
  q = double(S.qpos(1:nwin*win, k));
  v = double(S.qvel(1:nwin*win, k));
  qrms(:,k) = sqrt(mean(reshape(q, win, nwin).^2, 1))';
  vrms(:,k) = sqrt(mean(reshape(v, win, nwin).^2, 1))';
end
qrms = movmean(qrms, smoothWin, 1);
vrms = movmean(vrms, smoothWin, 1);
t_v  = ((1:nwin) - 0.5)*winSec/3600;

% ---------------------------------------------------
% Figure
% ---------------------------------------------------
xl    = [0 T/fs/3600];
xt    = linspace(0, xl(2), nTick);
xtl   = compose('%g', xt);
xtl(2:2:end) = {''};     % label 0, mid, end only: full labels collide across tiles
ncol  = nIa + 2;

fig = figure('Position', [100 100 fwidth fheight]);
tl = tiledlayout(nMN, ncol, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---- weight grid
for i = 1:nMN
  for j = 1:nIa
    ax = nexttile(tl, (i-1)*ncol + j);
    plot(ax, t_h, Wdec(:,i,j), Color=squeeze(cellCol(i,j,:))', LineWidth=LW);

    xlim(ax, xl);  ylim(ax, [0 1]);
    xticks(ax, xt);  yticks(ax, [0 0.5 1]);
    grid(ax, 'on');  box(ax, 'on');
    ax.FontSize = fontsize;  ax.Color = 'none';

    if j == 1
      yticklabels(ax, {0,'',1});
      ylabel(ax, labels{i}, 'FontWeight', 'bold');
    else
      yticklabels(ax, {});
    end
    if i == nMN
      xticklabels(ax, xtl);
    else
      xticklabels(ax, {});
    end
    if i == 1, title(ax, labels{j}, 'FontWeight', 'bold'); end
  end
end

% ---- pool motif column (one panel per pool, spanning its two rows)
for p = 1:(nMN/2)
  ax = nexttile(tl, (2*p-2)*ncol + nIa + 1, [2 1]);
  hold(ax, 'on');
  plot(ax, t_h, motif.rp(:,p), Color=shade(p)*cR, LineWidth=1.5*LW);
  plot(ax, t_h, motif.sp(:,p), Color=shade(p)*cS, LineWidth=1.5*LW);
  plot(ax, t_h, motif.ap(:,p), Color=shade(p)*cA, LineWidth=1.5*LW);

  xlim(ax, xl);  ylim(ax, [0 1]);
  xticks(ax, xt);  yticks(ax, [0 0.5 1]);  yticklabels(ax, {});
  grid(ax, 'on');  box(ax, 'on');
  ax.FontSize = fontsize;  ax.Color = 'none';
  if p == nMN/2, xticklabels(ax, xtl); else, xticklabels(ax, {}); end
  if p == 1, title(ax, 'Motifs', 'FontWeight', 'bold'); end
  text(ax, 0.04, 1.05, pool_lab{p}, 'Units', 'normalized', ...
       'FontWeight', 'bold', 'FontSize', fontsize);
  if p == 1
    lgMotif   = legend(ax, {'r','s','a'}, 'Location', 'east', 'Box', 'off', ...
                       'FontSize', fontsize-1);
    axMotif1  = ax;
  end
end

% ---- kinematics column (angle: top half, velocity: bottom half)
kin   = {qrms, vrms};
kinLb = {'RMS angle (rad)', 'RMS velocity (rad/s)'};
for k = 1:2
  ax = nexttile(tl, (k-1)*(nMN/2)*ncol + nIa + 2, [nMN/2 1]);
  hold(ax, 'on');
  for q = 1:size(kin{k},2)
    plot(ax, t_v, kin{k}(:,q), Color=cKin(q,:), LineWidth=LW);
  end
  xlim(ax, xl);  xticks(ax, xt);
  ylim(ax, [0 max(kin{k}(:))*1.1]);
  grid(ax, 'on');  box(ax, 'on');
  ax.FontSize = fontsize;  ax.Color = 'none';
  ax.YAxisLocation = 'right';
  ylabel(ax, kinLb{k});
  if k == 2, xticklabels(ax, xtl); else, xticklabels(ax, {}); end
  if k == 1
    title(ax, 'Kinematics', 'FontWeight', 'bold');
    legend(ax, {'Shoulder','Elbow'}, 'Location', 'northeast', 'Box', 'off', ...
           'FontSize', fontsize-1);
  end
end

xlabel(tl, 'Time (h)', 'FontSize', fontsize+1);
ylabel(tl, 'Motoneurons', 'FontSize', fontsize+1);
title(tl, 'Ia Neurons', 'FontSize', fontsize+1);

% Motif legend: keep it at the right edge ('east') but centred at legYFrac of
% the first motif panel's height instead of half. Done after the layout is
% final, since the tiled layout moves the axes as labels are added.
legYFrac = 0.75;
drawnow;
axMotif1.Units = 'normalized';  lgMotif.Units = 'normalized';
lgMotif.Position(2) = axMotif1.Position(2) + legYFrac*axMotif1.Position(4) ...
                      - lgMotif.Position(4)/2;

save_pdf(fig, fullfile(out_dir, 'grid.pdf'));
end
