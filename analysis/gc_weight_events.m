function [loc, pks, wij, dw, pruned] = gc_weight_events(W, i, j, opt)
%GC_WEIGHT_EVENTS  Detect weight-change events for connection (i,j).
%   Extracts the weight trace W(i,j) via GC_WTRACE (so W may be a cell
%   array or flat (m*m) x N storage), computes its positive rate of
%   change, finds peaks, gates them to the rising portion of the weight
%   (between 20% and 90% of max), and keeps the top opt.pct percent of
%   peaks by magnitude.
%
%   [loc, pks, wij, dw] = gc_weight_events(W, i, j, ...)
%     loc - sample indices of retained dw peaks
%     pks - dw_pos peak heights at loc
%     wij - full weight trace (1 x numel(W))
%     dw  - full weight derivative, per second (1 x numel(W))
%     pruned - true if the connection decays away, wij(end) < pruneFrac*max(wij).
%              The single pruning rule shared by gc_run and gc_event_grid.
%
%   Name-value options:
%     direction - which changes count as events:
%                 'both' (default) - peaks of |dw|, i.e. potentiation AND
%                                    depression. The value gate below is
%                                    direction-agnostic: it selects times when
%                                    the weight is mid-range, which is where
%                                    change happens whichever way it is going.
%                 'pos'            - peaks of the positive part of dw only
%                                    (potentiation; the pre-Aug-2026 behaviour)
%                 'neg'            - peaks of the negative part of dw only
%                                    (depression)
%     pct    - keep top pct percent of peaks by magnitude (default 100)
%     fs     - sampling rate in Hz, used to scale the derivative (default 200)
%     refPct - percentile of wij used as the reference for the gate; 100 = max
%              (default). A lower value (e.g. 99) makes the gate robust to a
%              single transient spike inflating the reference.
%     loFrac - gate lower bound = loFrac * reference (default 0.2)
%     hiFrac - gate upper bound = hiFrac * reference (default 0.9)
%     tWin   - time window as fractions of the run, [start end] in [0,1]
%              (default [0 1]; [0.5 1] keeps only the last half). The gate
%              reference is still taken over the FULL trace, so the window
%              selects when events occur without redefining the weight scale.
%     pruneFrac - pruning threshold for the pruned output (default 0.2)
%     nEvents - keep only the nEvents LARGEST peaks (after gate, tWin and pct;
%              default Inf = all). Equalizes event counts across connections
%              by magnitude, so small maintenance fluctuations drop out.
%     nEventsMode - how nEvents are chosen: 'largest' (default; the nEvents
%              largest peaks, as above) or 'random' (a uniform random subset of
%              nEvents, equalizing counts WITHOUT selecting by magnitude).
%     seed   - base seed for 'random'; the stream is seeded per connection
%              (seed*10000 + 100*(i-1) + j), so draws are reproducible.
%     dWrule - per-step weight updates from the learning rule, flat (m*m) x N
%              like W (dw_rule.bin, written by export_dw_rule.py; see
%              ds_load_dw_rule). When given, dw is computed from these instead
%              of differencing W, which is float32 and quantizes the per-step
%              increments. Same central-difference timing as gradient(wij):
%              dw(k) = (dW(k) + dW(k+1))/2/dt. Default [] = gradient(wij).
  arguments
    W
    i
    j
    opt.direction (1,:) char {mustBeMember(opt.direction,{'both','pos','neg'})} = 'both'
    opt.pct    (1,1) double = 100
    opt.fs     (1,1) double = 200
    opt.refPct (1,1) double = 100
    opt.loFrac (1,1) double = 0.2
    opt.hiFrac (1,1) double = 0.9
    opt.tWin   (1,2) double {mustBeInRange(opt.tWin,0,1)} = [0 1]
    opt.pruneFrac (1,1) double = 0.2
    opt.nEvents (1,1) double = Inf
    opt.dWrule = []
    opt.nEventsMode (1,:) char {mustBeMember(opt.nEventsMode,{'largest','random'})} = 'largest'
    opt.seed (1,1) double = 0
  end

  dt = 1/opt.fs;

  wij    = gc_wtrace(W, i, j);
  if isempty(opt.dWrule)
    dw = gradient(wij)/dt;
  else
    s  = gc_wtrace(opt.dWrule, i, j);       % s(k) = W(k) - W(k-1), unrounded
    dw = [s(2), (s(2:end-1) + s(3:end))/2, s(end)]/dt;
  end
  switch opt.direction
    case 'pos',  sig = (dw + abs(dw)) / 2;   % positive part
    case 'neg',  sig = (abs(dw) - dw) / 2;   % magnitude of the negative part
    case 'both', sig = abs(dw);              % either direction
  end
  [pks, loc] = findpeaks(sig);

  % Gate to the rising portion of the weight (reference = percentile of wij,
  % so a lone spike doesn't shift the band; refPct=100 reproduces max).
  ref = prctile(wij, opt.refPct);
  Wlo = opt.loFrac*ref;
  Whi = opt.hiFrac*ref;
  valid = wij(loc) > Wlo & wij(loc) < Whi;
  loc = loc(valid);
  pks = pks(valid);

  % Restrict to the requested fraction of the run
  T = numel(wij);
  inWin = loc >= opt.tWin(1)*T & loc <= opt.tWin(2)*T;
  loc = loc(inWin);
  pks = pks(inWin);

  % Keep top pct percent of peaks by magnitude
  [pks, si] = sort(pks, 'descend');
  loc = loc(si);
  thresh = prctile(pks, 100 - opt.pct);
  keep = pks >= thresh;
  pks = pks(keep);
  loc = loc(keep);

  % Keep nEvents: the largest (pks is already sorted descending) or a random
  % subset of the same size
  n = min(opt.nEvents, numel(pks));
  if strcmp(opt.nEventsMode, 'largest')
    pks = pks(1:n);
    loc = loc(1:n);
  else
    rs = RandStream('mt19937ar', 'Seed', opt.seed*10000 + (i-1)*100 + j);
    sel = sort(randperm(rs, numel(pks), n));
    pks = pks(sel);
    loc = loc(sel);
  end

  pruned = max(wij) <= 0 || wij(end) < opt.pruneFrac*max(wij);
end
