function R = gc_event_triggered_stack(S, W, conns, opt)
%GC_EVENT_TRIGGERED_STACK  Event-triggered signal panels, stacked per connection.
%
%   Generalises GC_EVENT_TRIGGERED_PLOT (1 connection) and
%   GC_EVENT_TRIGGERED_COMPARE (2 connections) to an arbitrary stack, and adds
%   a percentile BAND for event selection.
%
%   R = GC_EVENT_TRIGGERED_STACK(S, W, conns, ...)
%
%     conns  Nx2 [MN Ia] connection list -> one figure, N blocks stacked
%            'all'       -> six figures, one per MOTONEURON, each a 6x1 stack
%                           of that MN's six Ia inputs (MN is held fixed down
%                           the stack; the Ia source is what varies)
%            'diag'      -> the six reflex connections in one stack
%     R      struct array (one per figure) with .conns, .n_events, .fig, .file
%
%   printfig=true writes each figure with exportgraphics as
%   <printdir>/<printprefix>_<MN>.png, named for the motoneuron the stack
%   holds fixed -- so `conns='all'` yields gc_ets_SF.png ... gc_ets_BE.png.
%   A stack spanning several motoneurons is named for all of them.
%
%   Layout matches the 'compare' code: within each block, columns are signals
%   and rows are that signal's channels. The block for connection (i,j) has its
%   MN_i and Ia_j panels outlined in colour.
%
% ---------------------------------------------------------------------------
% EVENT SELECTION -- the `pct` interval
% ---------------------------------------------------------------------------
%   pct is in PERCENTILE units (0-100) of |dw| peak magnitude, matching the
%   existing scalar convention in GC_WEIGHT_EVENTS:
%
%     pct = 100        all gated events                       (default)
%     pct = 20         the top 20% by magnitude               (legacy scalar)
%     pct = [20 50]    events between the 20th and 50th pct   (NEW band)
%     pct = [0 50]     the weakest half
%
%   The scalar form is unchanged: `pct = P` means "top P percent", i.e. the
%   band [100-P, 100]. The two-element form is an explicit [lo hi] band, which
%   is what lets you ask whether small and large weight-change events have
%   different signal signatures -- something "top N%" cannot express.
%
%   Units are 0-100, NOT fractions. pct = [0.2 0.5] is legal but means the
%   0.2nd to 0.5th percentile (well under 1% of events); a warning fires if
%   both bounds are <= 1, in case fractions were intended.
%
%   Selection runs on top of GC_WEIGHT_EVENTS with pct=100, so the gating
%   (loFrac/hiFrac/refPct/direction) is exactly what the regression uses.
%
%   Example:
%     gc_event_triggered_stack(S, W, 'all', pct=[50 100], direction='pos');
%     gc_event_triggered_stack(S, W, [1 1; 5 1], pct=[0 50]);
%
%   See also GC_WEIGHT_EVENTS, GC_EVENT_GRID, GC_RUN.

  arguments
    S
    W
    conns = 'all'
    opt.pct (1,:) double = 100
    opt.direction (1,:) char {mustBeMember(opt.direction,{'both','pos','neg'})} = 'pos'
    opt.loFrac (1,1) double = 0
    opt.hiFrac (1,1) double = 1
    opt.refPct (1,1) double = 100
    opt.fs (1,1) double = 200
    opt.winSec (1,1) double = 1
    opt.alpha (1,1) double = 0.05
    opt.maxTraces (1,1) double = 300     % individual traces DRAWN per panel
    opt.signals (1,:) cell = {'qpos','qvel','drive','act_MN','act_Ia'}
    opt.labels  (1,:) cell = {'Position','Velocity','Babble','Motoneuron','Ia Neuron'}
    opt.units   (1,:) cell = {'(deg)','(deg/s)','','',''}
    opt.deg     (1,:) cell = {'qpos','qvel'}   % converted to degrees for display
    opt.ylims   (1,:) cell = {}          % {} = derived from the data
    opt.Position (1,:) double = []
    opt.muscles (1,:) cell = {'SF','SE','EF','EE','BF','BE'}
    opt.joints  (1,:) cell = {'S','E'}
    opt.printfig (1,1) logical = false     % write a .png per figure
    opt.printdir (1,:) char = './'
    opt.printprefix (1,:) char = 'gc_ets'
    opt.printres (1,1) double = 300
    opt.verbose (1,1) logical = true
  end

  [nMN, nIa] = gc_wsize(W);
  [lo_p, hi_p] = resolve_pct(opt.pct);

  % ---- which connections, and how they group into figures ----------------
  if ischar(conns) || isstring(conns)
    switch lower(string(conns))
      case "all"
        groups = arrayfun(@(i) [repmat(i,nIa,1), (1:nIa)'], 1:nMN, ...
                          'UniformOutput', false);
      case "diag"
        groups = {[(1:nMN)', (1:nMN)']};
      otherwise
        error('gc_event_triggered_stack:conns', ...
              'conns must be an Nx2 list, ''all'', or ''diag''.');
    end
  else
    validateattributes(conns, {'numeric'}, {'ncols',2,'positive','integer'});
    groups = {conns};
  end

  ylims = opt.ylims;
  if isempty(ylims), ylims = derive_ylims(S, opt); end

  R = struct('conns',{},'n_events',{},'fig',{},'file',{});
  for g = 1:numel(groups)
    R(g) = draw_stack(S, W, groups{g}, lo_p, hi_p, ylims, opt);
  end
end

% ===========================================================================
function [lo, hi] = resolve_pct(pct)
% Scalar -> legacy "top pct percent". Two elements -> explicit [lo hi] band.
  if isscalar(pct)
    validateattributes(pct, {'numeric'}, {'>=',0,'<=',100});
    lo = 100 - pct;  hi = 100;
  elseif numel(pct) == 2
    lo = pct(1);  hi = pct(2);
    if ~(lo >= 0 && hi <= 100 && lo < hi)
      error('gc_event_triggered_stack:pctBand', ...
        'pct band must satisfy 0 <= lo < hi <= 100; got [%g %g].', lo, hi);
    end
    if hi <= 1
      warning('gc_event_triggered_stack:pctUnits', ...
        ['pct = [%g %g] is in PERCENTILE units, so this keeps events between ' ...
         'the %gth and %gth percentile -- under 1%% of them. If you meant ' ...
         'fractions, use [%g %g].'], lo, hi, lo, hi, lo*100, hi*100);
    end
  else
    error('gc_event_triggered_stack:pct', ...
          'pct must be a scalar or a 2-element [lo hi] band.');
  end
end

% ===========================================================================
function ylims = derive_ylims(S, opt)
% Robust per-signal limits from the data itself. The hardcoded limits in the
% older 'compare' code clip ~70% of qpos on the current plant, which turns the
% Position row into traces running off-panel.
  ds = max(1, floor(size(S.qpos,1)/2e5));      % subsample; limits don't need 36M
  ylims = cell(1, numel(opt.signals));
  for q = 1:numel(opt.signals)
    x = double(S.(opt.signals{q})(1:ds:end, :));
    if any(strcmp(opt.signals{q}, opt.deg)), x = rad2deg(x); end
    lo = prctile(x(:), 0.1);  hi = prctile(x(:), 99.9);
    if lo < 0 && hi > 0                 % signed -> symmetric about zero
      m = max(abs([lo hi]));  lo = -m;  hi = m;
    elseif lo >= 0
      lo = 0;                           % non-negative -> anchor at 0
    end
    if hi <= lo, hi = lo + 1; end
    ylims{q} = [-nice_floor(-lo), nice_floor(hi)];
    if ylims{q}(2) <= ylims{q}(1), ylims{q} = [lo hi]; end
  end
end

% ===========================================================================
function v = nice_floor(x)
% Largest "nice" value <= x: mantissa floored to a half step, so limits read as
% 35 / 80 / 1 / 0.35 rather than 36.5 / 84.8 / 1.03 / 0.391. The inputs are
% already 0.1/99.9 percentiles, so there is headroom and flooring does not
% visibly clip.
  if ~isfinite(x) || x <= 0, v = 0; return; end
  e = floor(log10(x));
  m = x / 10^e;                    % mantissa in [1,10)
  v = floor(m*2)/2 * 10^e;
end

% ===========================================================================
function r = draw_stack(S, W, cc, lo_p, hi_p, ylims, opt)
  fs = opt.fs;  dt = 1/fs;
  halfwin = round((opt.winSec/2)/dt);
  win = -halfwin:halfwin;
  N = size(S.qpos,1);
  nB = size(cc,1);
  nq = numel(opt.signals);

  % ---- events per block --------------------------------------------------
  locs = cell(nB,1);
  for b = 1:nB
    [loc, pks] = gc_weight_events(W, cc(b,1), cc(b,2), ...
        pct=100, fs=fs, refPct=opt.refPct, ...
        loFrac=opt.loFrac, hiFrac=opt.hiFrac, direction=opt.direction);
    if ~isempty(loc)
      % percentile BAND on peak magnitude
      tlo = prctile(pks, lo_p);  thi = prctile(pks, hi_p);
      % half-open [tlo, thi) so [0 50] and [50 100] partition the events
      % instead of both claiming the ties sitting exactly at the median;
      % the top of the range stays closed so pct=[x 100] keeps the maximum.
      if hi_p >= 100
        keep = pks >= tlo & pks <= thi;
      else
        keep = pks >= tlo & pks <  thi;
      end
      loc = loc(keep);
      % drop events whose window would run off either end
      loc = loc(loc > halfwin & loc <= N - halfwin);
      loc = sort(loc);          % gc_weight_events returns peak-order, not time
    end
    locs{b} = loc;
  end

  if opt.verbose
    % One count per BLOCK (connection). Every signal panel inside a block
    % shares it: events are detected on the weight trace w_ij alone, and the
    % signals are only what gets averaged around them.
    fprintf('stack: pct band [%g %g]\n', lo_p, hi_p);
    for b = 1:nB
      fprintf('  %s<-%s  n = %d\n', opt.muscles{cc(b,1)}, ...
              opt.muscles{cc(b,2)}, numel(locs{b}));
    end
  end

  % ---- geometry (same construction as the 'compare' layout) --------------
  % Tuned for a narrow publication aspect (~373 pt wide). Font sizes are in
  % POINTS, so they do not shrink with the figure -- the margins below are what
  % keeps 10 pt labels from colliding at this width.
  xsp = 0.050;  ysp = 0.004;  ygap = 0.016;
  xgut = 0.088;              % left gutter: block label + n, vertically centred
  xlab = 0.140;              % plot area starts here (gutter + y ticks + units)
  xmargin = 0.050;           % right margin: the right-justified muscle names
  topM = 0.072;              % title + one line of column headers
  botM = 0.050;              % time axis label
  w = (1 - xlab - xmargin - (nq-1)*xsp)/nq;
  bandH = (1 - topM - botM - (nB-1)*ygap)/nB;

  pos = opt.Position;
  if isempty(pos), pos = [80 40 373 min(1500, 103*nB)]; end
  fig = figure('Color','w','Position',pos);

  for b = 1:nB
    i = cc(b,1);  j = cc(b,2);
    loc = locs{b};
    ytop = 1 - topM - (b-1)*(bandH + ygap);
    xs = xlab;

    for q = 1:nq
      sig = opt.signals{q};
      X = S.(sig);
      nsig = size(X,2);
      h = (bandH - (nsig-1)*ysp)/nsig;
      ys = ytop - h;

      for n = 1:nsig
        ax = axes('Position',[xs ys w h]); hold(ax,'on');
        [tr, mu] = event_windows(X(:,n), loc, win, opt, ...
                                 any(strcmp(sig,opt.deg)));
        if ~isempty(mu)
          plot(ax, win*dt*1000, tr, 'Color', [0 0 0 opt.alpha]);
          plot(ax, win*dt*1000, mu, 'r', 'LineWidth', 1);
        end
        ylim(ax, ylims{q});
        plot(ax, [0 0], ylims{q}, '--k', 'LineWidth', 0.5);

        ax.Color = 'none';  box(ax,'off');  grid(ax,'off');
        ax.TickDir = 'out';  ax.FontSize = 7;
        yl_ = ylims{q};
        yticks(ax, linspace(yl_(1), yl_(2), 3));
        yticklabels(ax, {num2str(yl_(1)), '', num2str(yl_(2))});
        if n ~= nsig, yticklabels(ax, {}); end

        % --- marker for the two channels this connection uses ---------------
        % Template construction: LineStyle 'none' drops the shaft and leaves a
        % solid black arrowhead in the gutter -- higher contrast than a tint.
        if (strcmp(sig,'act_Ia') && n==j) || (strcmp(sig,'act_MN') && n==i)
          ar = annotation('arrow','Position',[xs-0.018, ys+h/2, 0.009, 0]);
          ar.LineStyle = 'none';
          ar.HeadLength = 6;  ar.HeadWidth = 6;
        end

        % --- muscle names, last column, right-justified ---------------------
        if q == nq && nsig == numel(opt.muscles)
          ann = annotation('textbox','Position',[xs+w+0.004, ys, xmargin-0.006, h]);
          ann.String = opt.muscles{n};
          ann.Interpreter = 'none';
          ann.EdgeColor = 'none';
          ann.FontSize = 10;
          ann.HorizontalAlignment = 'Left';
          ann.VerticalAlignment = 'Middle';
        end

        % --- joint + flexion sense, bottom block only -----------------------
        if b == nB && nsig == numel(opt.joints)
          if n == 1
            ann = annotation('textbox','Position',[xs, ys, w, h]);
            ann.String = opt.joints{1};
            ann.Interpreter = 'none';
            ann.EdgeColor = 'none';
            ann.FontSize = 10;
            ann.HorizontalAlignment = 'Left';
            ann.VerticalAlignment = 'Top';
          elseif n == 2
            ann = annotation('textbox','Position',[xs, ys, w, h]);
            ann.String = opt.joints{2};
            ann.Interpreter = 'none';
            ann.EdgeColor = 'none';
            ann.FontSize = 10;

            ann = annotation('textbox','Position',[xs+0.46*w, ys, 0.5*w, h]);
            ann.String = '\uparrow Flx.';
            ann.EdgeColor = 'none';
            ann.VerticalAlignment = 'top';
            ann.FontSize = 8;
            ann.FitBoxToText = 'on';   % never wrap inside a narrow column
            ann.Margin = 0;

            ann = annotation('textbox','Position',[xs+0.46*w, ys, 0.5*w, h]);
            ann.String = '\downarrow Ext.';
            ann.EdgeColor = 'none';
            ann.VerticalAlignment = 'bottom';
            ann.FontSize = 8;
            ann.FitBoxToText = 'on';
            ann.Margin = 0;
          end
        end

        % time axis only on the very bottom row of the figure
        if b == nB && q == ceil(nq/2) && n == nsig
          xlabel(ax, 'Time from weight-change event (ms)', 'FontSize',8);
        else
          xticklabels(ax, {});
        end
        if n == nsig && ~isempty(opt.units{q})
          ylab = ylabel(ax, opt.units{q}, 'FontSize',8);
          ylab.Units = 'normalized';
          ylab.Position(1) = ylab.Position(1) + 0.11;
        end

        % --- signal name, top block, template annotation --------------------
        if b == 1 && n == 1
          ann = annotation('textbox','Position',[xs, ytop+0.006, w, 0.024]);
          ann.String = opt.labels{q};
          ann.Interpreter = 'none';
          ann.EdgeColor = 'none';
          ann.FontSize = 10;
          ann.FitBoxToText = 'on';     % never wrap: the column is narrower
          ann.VerticalAlignment = 'middle';
        end
        ys = ys - ysp - h;
      end
      xs = xs + xsp + w;
    end

    % block label on the left
    % Block identity sits in the top-left of its own first panel (under the
    % 'Position' header on the top block), which frees the left gutter.
    yc = ytop - bandH/2;                       % vertical centre of the block
    annotation('textbox', [0.004, yc + 0.000, xgut, 0.016], ...
      'String', sprintf('%s\\leftarrow%s', opt.muscles{i}, opt.muscles{j}), ...
      'EdgeColor','none', 'FontWeight','bold', 'FontSize',8, ...
      'HorizontalAlignment','left', 'VerticalAlignment','bottom', ...
      'Interpreter','tex', 'Margin',0);
    annotation('textbox', [0.004, yc - 0.014, xgut, 0.014], ...
      'String', sprintf('n = %d', numel(loc)), ...
      'EdgeColor','none', 'FontSize',7, 'Color',[0.25 0.25 0.25], ...
      'HorizontalAlignment','left', 'VerticalAlignment','top', 'Margin',0);
  end

  % An annotation, not sgtitle: with manually positioned axes sgtitle does not
  % reserve space and lands on top of the first block's panels.
  annotation('textbox', [0, 0.962, 1, 0.034], 'String', ...
      sprintf(['Event-triggered signals  |  dir=%s,  pct band [%g %g],  ' ...
               'gate [%.2f %.2f]\\timesp%g,  %g ms window'], ...
      opt.direction, lo_p, hi_p, opt.loFrac, opt.hiFrac, opt.refPct, ...
      opt.winSec*1000), ...
      'EdgeColor','none', 'FontSize',10, 'Interpreter','tex', ...
      'HorizontalAlignment','center', 'VerticalAlignment','middle');

  % ---- optional .png, named for the motoneuron held fixed ----------------
  fname = '';
  if opt.printfig
    mns = unique(cc(:,1))';
    tag = strjoin(arrayfun(@(k) opt.muscles{k}, mns, 'UniformOutput', false), '-');
    if ~isfolder(opt.printdir), mkdir(opt.printdir); end
    fname = fullfile(opt.printdir, ...
                     sprintf('%s_%s.png', opt.printprefix, tag));
    exportgraphics(fig, fname, 'Resolution', opt.printres);
    if opt.verbose, fprintf('  saved %s\n', fname); end
  end

  r = struct('conns', cc, 'n_events', cellfun(@numel, locs), 'fig', fig, ...
             'file', fname);
end

% ===========================================================================
function [tr, mu] = event_windows(x, loc, win, opt, to_deg)
% Mean over ALL events, but only a random subset is returned for drawing.
% At ~1e5 events a full x(win+loc') is gigabytes and would draw 1e5 lines per
% panel; the mean is accumulated in chunks instead.
  tr = [];  mu = [];
  if isempty(loc), return; end
  x = double(x);
  if to_deg, x = rad2deg(x); end
  nW = numel(win);  nE = numel(loc);

  acc = zeros(1, nW);
  chunk = 5000;
  for a = 1:chunk:nE
    idx = loc(a:min(a+chunk-1, nE));
    acc = acc + sum(x(idx(:) + win), 1);
  end
  mu = acc / nE;

  nDraw = min(opt.maxTraces, nE);
  sel = loc(round(linspace(1, nE, nDraw)));    % even spread over the session
  tr = x(sel(:) + win)';
end
