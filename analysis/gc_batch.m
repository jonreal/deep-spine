function R = gc_batch(S,W,opt)
%GC_BATCH  Granger-style kinematic test for every Ia->MN connection.
%   Runs gc_run on all nMN x nIa connections using potentiation ('pos') events
%   only. The gating options (pct, refPct, loFrac, hiFrac, tWin, pruneFrac) are
%   set here once and forwarded to both gc_run and gc_event_grid, so the event
%   grid shows exactly the events the regression uses.
%
%   Antagonist connections (flexor <-> extensor, opposite-parity cells) are
%   not tested by default (skipAntagonist=true): they all decay, so they have
%   no growth phase, and the hypothesis compares reflex with synergist
%   connections. Their cells are NaN (blank), like pruned connections.
%
%   Final figure: nested-F statistic and -log10(p) for the kinematic block
%   beyond the Ia + dw-history baseline. All other statistics are kept in R
%   and R.table.
%
%   R = gc_batch(S, W, loFrac=0, hiFrac=0.9, pct=100, tWin=[0.5 1], ...)
  arguments
    S
    W
    opt.baseline (1,:) char {mustBeMember(opt.baseline,{'Ia','MN','both'})} = 'Ia'
    opt.detrend (1,:) char {mustBeMember(opt.detrend,{'none','diff','highpass'})} = 'none'
    opt.hpCutoff (1,1) double = 0.5
    opt.pct (1,1) double = 100
    opt.refPct (1,1) double = 100
    opt.loFrac (1,1) double = 0
    opt.hiFrac (1,1) double = 0.9
    opt.tWin (1,2) double = [0 1]          % fraction of the run, e.g. [0.5 1]
    opt.pruneFrac (1,1) double = 0.2
    opt.nEvents (1,1) double = Inf       % keep the N largest events per connection
    opt.dWrule = []                      % rule-computed per-step dW (gc_weight_events)
    opt.nEventsMode (1,:) char = 'largest'   % 'largest' or 'random' nEvents (gc_weight_events)
    opt.seed (1,1) double = 0            % base seed for nEventsMode='random'
    opt.skipAntagonist (1,1) logical = true   % leave flexor<->extensor cells untested
    opt.alpha (1,1) double = 0.025
    opt.winSec (1,1) double = 1
    opt.lag (1,1) double = 0.25
    opt.feature = 'full';
    opt.plot (1,1) logical = true
    opt.grid (1,1) logical = true          % draw gc_event_grid with the same gating
    opt.pcap (1,1) double = 6              % display cap for -log10(p)
    opt.marks (1,1) logical = true         % star SF<-SF, triangle BF<-SF
    % light grey -> near black: orthogonal to the parula motif colors (Fig 2),
    % magma (Fig 5) and blue-white-red (W); starts off-white so a zero cell
    % stays distinct from a white (NaN, pruned) cell
    opt.cmap = linspace(0.92, 0.05, 256)' * [1 1 1]
  end

  nMN = size(S.act_MN,2);
  nIa = size(S.act_Ia,2);
  dR2_MN       = nan(nMN,nIa);
  dR2_Ia       = nan(nMN,nIa);
  partialR2_MN = nan(nMN,nIa);
  partialR2_Ia = nan(nMN,nIa);
  dAIC_MN      = nan(nMN,nIa);
  dAIC_Ia      = nan(nMN,nIa);
  dBIC_MN      = nan(nMN,nIa);
  dBIC_Ia      = nan(nMN,nIa);
  F_MN         = nan(nMN,nIa);
  F_Ia         = nan(nMN,nIa);
  p_MN         = nan(nMN,nIa);
  p_Ia         = nan(nMN,nIa);
  w_init       = nan(nMN,nIa);
  w_final      = nan(nMN,nIa);
  w_net        = nan(nMN,nIa);

  for i = 1:nMN
    for j = 1:nIa
      fprintf('Running gc_run i=%d, j=%d\n', i, j);
      % Weight fate is recorded for every connection, including those gc_run
      % prunes, so the table shows why a cell came back NaN.
      wtr = gc_wtrace(W, i, j);
      w_init(i,j)  = wtr(1);
      w_final(i,j) = wtr(end);
      w_net(i,j)   = wtr(end) - wtr(1);
      if opt.skipAntagonist && mod(i+j,2) == 1
        continue                              % antagonist: untested, stays NaN
      end
      try
        [dR2_MN(i,j), dR2_Ia(i,j), ...
          partialR2_MN(i,j), partialR2_Ia(i,j), ...
          dAIC_MN(i,j), dAIC_Ia(i,j), dBIC_MN(i,j), dBIC_Ia(i,j), ...
          F_MN(i,j), F_Ia(i,j), p_MN(i,j), p_Ia(i,j)] = gc_run( ...
            S,W,i,j, ...
            direction='pos', ...
            baseline=opt.baseline, ...
            detrend=opt.detrend, ...
            hpCutoff=opt.hpCutoff, ...
            pct=opt.pct, ...
            loFrac=opt.loFrac, ...
            hiFrac=opt.hiFrac, ...
            refPct=opt.refPct, ...
            tWin=opt.tWin, ...
            pruneFrac=opt.pruneFrac, ...
            nEvents=opt.nEvents, ...
            dWrule=opt.dWrule, ...
            nEventsMode=opt.nEventsMode, ...
            seed=opt.seed, ...
            alpha=opt.alpha, ...
            winSec=opt.winSec, ...
            feature=opt.feature, ...
            lag=opt.lag);

      catch ME
        warning('gc_run failed for i=%d, j=%d: %s', i, j, ME.message);
      end
    end
  end

  R.dR2_MN       = dR2_MN;
  R.dR2_Ia       = dR2_Ia;
  R.partialR2_MN = partialR2_MN;
  R.partialR2_Ia = partialR2_Ia;
  R.dAIC_MN      = dAIC_MN;
  R.dAIC_Ia      = dAIC_Ia;
  R.dBIC_MN      = dBIC_MN;
  R.dBIC_Ia      = dBIC_Ia;
  R.F_MN         = F_MN;
  R.F_Ia         = F_Ia;
  R.p_MN         = p_MN;
  R.p_Ia         = p_Ia;
  R.w_init       = w_init;
  R.w_final      = w_final;
  R.w_net        = w_net;

  labels = {'SF','SE','EF','EE','BF','BE'};

  if opt.grid
    R.grid = gc_event_grid(W, direction='pos', pct=opt.pct, ...
      refPct=opt.refPct, loFrac=opt.loFrac, hiFrac=opt.hiFrac, ...
      tWin=opt.tWin, pruneFrac=opt.pruneFrac, nEvents=opt.nEvents, labels=labels);
  end

  % -log10(p) is capped for display. p-values here are anti-conservative
  % (autocorrelated, peak-conditioned) so gradation past the cap is not
  % meaningful, and a single underflowed cell at ~308 would otherwise crush
  % everything else to the bottom of the colour scale.
  if opt.plot
    figure('Color','w');
    subplot(1,2,1); draw_panel(F_Ia, 'F: kin | Ia + history', labels, opt.marks, [], opt.cmap);
    subplot(1,2,2); draw_panel(neglog10p(p_Ia), '-log_{10}(p): kin | Ia + history', ...
                               labels, opt.marks, [0 opt.pcap], opt.cmap);
    sgtitle(sprintf(['Nested F-test (pos events), N=%g largest, gate [%.2g, %.2g], pct=%g, ' ...
        't\\in[%.2g, %.2g], lag=%.3g s, -log_{10}p capped at %g'], ...
        opt.nEvents, opt.loFrac, opt.hiFrac, opt.pct, opt.tWin(1), opt.tWin(2), opt.lag, opt.pcap));
  end

T = gc_connection_table(dR2_MN, dR2_Ia, partialR2_MN, partialR2_Ia, ...
      dAIC_MN, dAIC_Ia, dBIC_MN, dBIC_Ia, F_MN, F_Ia, p_MN, p_Ia, ...
      w_init, w_final, w_net, labels);
R.table = T;
disp(sortrows(T,'dR2_Ia','descend'));

end

function draw_panel(M, ttl, labels, marks, climIn, cmap)
% imagesc with muscle-name ticks. Pruned (NaN) cells render WHITE, not as the
% bottom of the colour scale -- otherwise "this connection decayed away" and
% "this connection has no effect" look identical. If marks=true, flag the two
% example connections: star = (i,j)=(1,1), triangle = (5,1). climIn optionally
% fixes the colour limits.
  if nargin < 4, marks = false; end
  if nargin < 5, climIn = []; end
  if nargin < 6, cmap = []; end
  bad = isnan(M);
  M(bad) = 0;
  him = imagesc(M);
  set(him, 'AlphaData', ~bad);          % NaN cells show the axes background
  set(gca, 'Color', [1 1 1]);
  if ~isempty(climIn) && diff(climIn) > 0, clim(climIn); end
  if ~isempty(cmap), colormap(gca, cmap); end
  axis image; colorbar;
  xticks(1:size(M,2)); yticks(1:size(M,1));
  xticklabels(labels); yticklabels(labels);
  xlabel('Ia Neurons'); ylabel('Motoneurons');
  title(ttl);
  if marks
    hold on;
    % imagesc: x = column (Ia, j), y = row (MN, i)
    plot(1, 1, 'p', 'MarkerFaceColor','w', 'MarkerEdgeColor','none', 'MarkerSize',12); % SF<-SF
    plot(1, 5, '^', 'MarkerFaceColor','w', 'MarkerEdgeColor','none', 'MarkerSize',9);  % BF<-SF
    hold off;
  end
end

function print_gc_matrix(name,A)
  fprintf('\n%s\n', name);
  disp(array2table(A, ...
      'VariableNames', compose('Ia%d',1:size(A,2)), ...
      'RowNames', compose('MN%d',1:size(A,1))));
end


function L = neglog10p(P)
% -log10(p) with p floored at realmin. At large event counts the nested F is
% big enough that fcdf underflows to exactly 0, and -log10(0) = Inf would blow
% up the colour scale (draw_panel zeroes NaN but not Inf). The floor caps the
% panel at ~308 instead. A saturated panel is itself the signal that p has
% stopped discriminating -- see gc-readme.md.
  L = -log10(max(P, realmin));
  L(isnan(P)) = NaN;   % max() ignores NaN, so pruned cells would otherwise
                       % come back fully saturated instead of blank
end

function T = gc_connection_table(dR2_MN,dR2_Ia,pR2_MN,pR2_Ia, ...
                                 dAIC_MN,dAIC_Ia,dBIC_MN,dBIC_Ia, ...
                                 F_MN,F_Ia,p_MN,p_Ia, ...
                                 w_init,w_final,w_net,labels)

  [I,J] = ndgrid(1:size(dR2_MN,1), 1:size(dR2_MN,2));
  conn = strings(numel(I),1);
  for n = 1:numel(I)
    conn(n) = labels{I(n)} + "←" + labels{J(n)};
  end
  T = table(I(:), J(:), conn, ...
      dR2_MN(:), dR2_Ia(:), pR2_MN(:), pR2_Ia(:), ...
      dAIC_MN(:), dAIC_Ia(:), dBIC_MN(:), dBIC_Ia(:), ...
      F_MN(:), F_Ia(:), p_MN(:), p_Ia(:), ...
      w_init(:), w_final(:), w_net(:), ...
      'VariableNames', {'MN','Ia','Connection', ...
      'dR2_MN','dR2_Ia','partialR2_MN','partialR2_Ia', ...
      'dAIC_MN','dAIC_Ia','dBIC_MN','dBIC_Ia', ...
      'F_MN','F_Ia','p_MN','p_Ia', ...
      'w_init','w_final','w_net'});

end
