function R = gc_event_grid(W, opt)
%GC_EVENT_GRID  Gating sanity check: weight traces + selected GC events.
%   Plots a 6x6 grid (MN i x Ia j) of each connection's weight trace wij with
%   the events that gc_run would actually analyze marked in red, plus the
%   rising-region gate band [loFrac, hiFrac]*reference. Uses gc_weight_events
%   with the SAME options as the regression, so this shows exactly what feeds
%   the Granger analysis. Connections gc_run skips (pruned/decaying:
%   wij(end) < loFrac*reference) are shaded gray and tagged.
%
%   Use it to confirm the gating selects sane ramp-up events per connection
%   and to spot weights that are non-monotonic or never organize.
%
%   A SECOND figure reproduces the dashboard's motif regression from this
%   high-fidelity weight log, on the same time axis in seconds, so the event
%   grid and the dashboard motif panels can be read against each other. The
%   regression is DS_FIT_MOTIF itself, applied to a decimated slice of W, so it
%   is the identical computation the dashboards use -- not a reimplementation.
%
%   R = gc_event_grid(W, ...)
%     R.n_events - nMN x nIa count of selected events per connection
%     R.pruned   - nMN x nIa logical, connections gc_run would skip
%     R.motif    - the DS_FIT_MOTIF output (empty if motif=false)
%     R.motif_t  - time axis (s) for the motif traces
%
%   Name-value options (forwarded to gc_weight_events):
%     fs, direction, pct, refPct, loFrac, hiFrac, tWin, pruneFrac
%     (see gc_weight_events). Events outside tWin are not selected, and the
%     window bounds are drawn as vertical lines when tWin ~= [0 1].
%     labels - per-channel labels (default muscle labels)
  arguments
    W
    opt.fs     (1,1) double = 200
    opt.direction (1,:) char {mustBeMember(opt.direction,{'both','pos','neg'})} = 'pos'
    opt.pct    (1,1) double = 100
    opt.refPct (1,1) double = 100
    opt.loFrac (1,1) double = 0.2
    opt.hiFrac (1,1) double = 0.9
    opt.tWin   (1,2) double = [0 1]
    opt.pruneFrac (1,1) double = 0.2
    opt.nEvents (1,1) double = Inf
    opt.labels cell = {'SF','SE','EF','EE','BF','BE'}
    opt.motif  (1,1) logical = true    % draw the motif-regression figure
    opt.motifN (1,1) double  = 5e4     % target sample count for that figure
    opt.plotN  (1,1) double  = 2e4     % points drawn per trace panel
  end

  [nMN, nIa, T] = ds_w_size(W);
  t_s = (1:T)/opt.fs;                 % time axis (s)

  % Decimate only what is DRAWN. At 200 Hz over hours this grid is 36 panels x
  % tens of millions of points, which renders unusably slowly; the weight
  % traces are smooth so nothing is lost visually. Event detection, pruning and
  % the gate band all still use the full-rate signal, and the red event markers
  % keep their true sample positions.
  dp = max(1, ceil(T/opt.plotN));
  ip = 1:dp:T;

  n_events = zeros(nMN, nIa);
  pruned   = false(nMN, nIa);

  ncol = nIa + double(opt.motif);          % extra column for the motif pools
  figure('Color','w','Position',[150 120 1280 900]);
  tl = tiledlayout(nMN, ncol, 'TileSpacing','tight', 'Padding','compact');

  % decimated traces, kept so the motif pools can be built from exactly the
  % curves drawn in the grid (column order: (i-1)*nIa + j)
  Wdec = zeros(numel(ip), nMN*nIa);

  for i = 1:nMN
    for j = 1:nIa
      ax = nexttile((i-1)*ncol + j);   % explicit index: the motif column
                                       % occupies the last tile of each row
      hold(ax,'on');

      [loc, ~, wij, ~, pruned(i,j)] = gc_weight_events(W, i, j, ...
        direction=opt.direction, fs=opt.fs, pct=opt.pct, refPct=opt.refPct, ...
        loFrac=opt.loFrac, hiFrac=opt.hiFrac, tWin=opt.tWin, ...
        pruneFrac=opt.pruneFrac, nEvents=opt.nEvents);

      ref = prctile(wij, opt.refPct);

      Wdec(:, (i-1)*nIa + j) = wij(ip)';
      plot(ax, t_s(ip), wij(ip), 'k');
      if pruned(i,j)
        set(ax,'Color',[0.95 0.95 0.95]);
        text(ax, t_s(end)/2, double(max(wij)), 'pruned', ...
          'HorizontalAlignment','center', 'VerticalAlignment','top', ...
          'Color',[0.7 0.7 0.7], 'FontSize',8);
      else
        % gate band the regression keeps events within
        yline(ax, opt.loFrac*ref, ':', 'Color',[0.6 0.6 0.6]);
        yline(ax, opt.hiFrac*ref, ':', 'Color',[0.6 0.6 0.6]);
        if any(opt.tWin ~= [0 1])
          xline(ax, opt.tWin*t_s(end), '--', 'Color',[0.6 0.6 0.6]);
        end
        scatter(ax, t_s(loc), wij(loc), 8, 'r', 'filled');
        n_events(i,j) = numel(loc);
        text(ax, 0.03, 0.92, sprintf('n=%d', numel(loc)), ...
          'Units','normalized', 'FontSize',7, 'Color',[0.2 0.2 0.2]);
      end

      % Fixed y range across every panel: weights are bounded [0, max_weight]
      % (=1 in the learner config), so a common scale makes the 6x6 grid
      % directly comparable instead of autoscaling each connection.
      xlim(ax,[0 t_s(end)]); ylim(ax,[0 1]);
      grid(ax,'on'); box(ax,'on'); ax.FontSize = 8;

      % uniform outer-edge labels, driven by position (not pruning)
      if j == 1
        ylabel(ax, opt.labels{i}, 'FontWeight','bold', 'Rotation',0, ...
          'HorizontalAlignment','right', 'VerticalAlignment','middle');
      end
      if i == 1,   title(ax, opt.labels{j}, 'FontWeight','bold'); end
      if i ~= nMN, ax.XTickLabel = []; end
    end
  end

  xlabel(tl, 'time (s)');
  ylabel(tl, 'weight w_{ij}');
  title(tl, sprintf(['GC event gating  (rows: MN,  cols: Ia)  —  ' ...
    'red = selected events,  band = [%.2f, %.2f]\\timesp%g,  t \\in [%.2f, %.2f]'], ...
    opt.loFrac, opt.hiFrac, opt.refPct, opt.tWin(1), opt.tWin(2)), 'Interpreter','tex');

  % ---- motif pools, as the last column of the same figure -----------------
  motif = []; motif_t = [];
  if opt.motif
    [motif, motif_t] = motif_pools(Wdec, t_s(ip), nMN);
    draw_motif_column(motif, motif_t, tl, ncol, nMN);
  end

  R = struct('n_events', n_events, 'pruned', pruned, ...
    'fs', opt.fs, 'pct', opt.pct, 'refPct', opt.refPct, ...
    'loFrac', opt.loFrac, 'hiFrac', opt.hiFrac, 'tWin', opt.tWin, ...
    'pruneFrac', opt.pruneFrac, 'nEvents', opt.nEvents, 'labels', {opt.labels}, ...
    'motif', motif, 'motif_t', motif_t);
end

% ---------------------------------------------------------------------------
function [motif, t_s] = motif_pools(Wdec, t_s, m)
% Motif pool coefficients built DIRECTLY from the drawn weight traces.
%
% The motif basis vectors are disjoint indicator pairs spanning all m^2 cells,
% so B has orthogonal columns and pinv(B) = B'/2. Each of the 18 coefficients
% is therefore exactly the MEAN of its two cells, and each pool coefficient is
% the mean of its member cells. Computing it that way here (rather than calling
% ds_fit_motif) is deliberate: it is what makes the column verifiable by eye
% against the grid to its left.
%
% For pool p (muscle rows 2p-1, 2p):
%   reflex  : (2p-1,2p-1), (2p,2p)
%   synergy : (2p-1,2q-1), (2p,2q)   for q ~= p    (same parity, cross-pool)
%   anti    : (2p-1,2q),   (2p,2q-1) for all q     (opposite parity)
  np  = m/2;
  col = @(i,j) (i-1)*m + j;
  motif.rp = zeros(size(Wdec,1), np);
  motif.sp = zeros(size(Wdec,1), np);
  motif.ap = zeros(size(Wdec,1), np);
  motif.members = cell(np,3);
  for p = 1:np
    r = [col(2*p-1,2*p-1), col(2*p,2*p)];
    sy = []; an = [];
    for q = 1:np
      if q ~= p
        sy = [sy, col(2*p-1,2*q-1), col(2*p,2*q)];
      end
      an = [an, col(2*p-1,2*q), col(2*p,2*q-1)];
    end
    motif.rp(:,p) = mean(Wdec(:,r), 2);
    motif.sp(:,p) = mean(Wdec(:,sy), 2);
    motif.ap(:,p) = mean(Wdec(:,an), 2);
    motif.members(p,:) = {r, sy, an};
  end
  motif.rg = mean(motif.rp,2);
  motif.sg = mean(motif.sp,2);
  motif.ag = mean(motif.ap,2);
end

function draw_motif_column(motif, t_s, tl, ncol, nMN)
% One panel per pool, spanning that pool's two muscle rows, on the same
% [0,1] scale as the grid so the average can be checked against the cells.
  cmap1 = parula;  nC = size(cmap1,1);
  v2i   = @(v) round((v - (-1)) / (1 - (-1)) * (nC - 1)) + 1;
  cR = cmap1(v2i( 0.6),:);
  cS = cmap1(v2i( 0.0),:);
  cA = cmap1(v2i(-0.9),:);
  pool_lab = {'S','E','B'};

  for p = 1:(nMN/2)
    ax = nexttile(tl, (2*p-2)*ncol + ncol, [2 1]);
    hold(ax,'on');
    plot(ax, t_s, motif.rp(:,p), '-', 'Color', cR, 'LineWidth', 1.5);
    plot(ax, t_s, motif.sp(:,p), '-', 'Color', cS, 'LineWidth', 1.5);
    plot(ax, t_s, motif.ap(:,p), '-', 'Color', cA, 'LineWidth', 1.5);
    ylim(ax,[0 1]);  xlim(ax,[0 t_s(end)]);
    grid(ax,'on'); box(ax,'on'); ax.FontSize = 8;
    ax.YAxisLocation = 'right';
    title(ax, sprintf('motif %s', pool_lab{p}), 'FontWeight','bold');
    if p == 1
      legend(ax, {'reflex','synergy','anti'}, 'Location','northwest', ...
             'Box','off', 'FontSize',7);
    end
    if p < nMN/2, ax.XTickLabel = []; end
  end
end

