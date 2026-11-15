function R = gc_histogram(W, opt)
%GC_HISTOGRAM  6x6 grid of event-magnitude distributions, one panel per W(i,j).
%
%   R = GC_HISTOGRAM(W, ...) draws, for every connection, the distribution of
%   |dw| peak magnitudes over the events GC_WEIGHT_EVENTS selects. Layout
%   matches GC_EVENT_GRID (rows = motoneuron, cols = Ia neuron). One figure per
%   direction in `directions` (default: a 'pos' figure and a 'neg' figure).
%
%   dw is QUANTIZED: hebbian.txt stores weights as float32, so dw moves in
%   steps of eps(single(w))*fs/2 -- and eps doubles at each power of two, so W
%   in [0.125,0.25), [0.25,0.5) and [0.5,1) sit on lattices of 1x, 2x and 4x
%   the base quantum. Binning must respect that: if the edges fall unevenly
%   across lattice positions you get moire, alternating tall and short bins
%   that look like structure and are not.
%
%   So bin width is a whole number of quanta and the edges are offset by half
%   a quantum, putting the same number of lattice positions in every bin. x is
%   in quanta by default (xunits='quantum'), which makes the levels integers
%   and the panels comparable; 'raw' keeps the original units. All panels share
%   one x range, as in GC_EVENT_GRID.
%
%   That discreteness is what makes a narrow `pct` band come back empty: with
%   u distinct values, one tied level can span several percentiles, so
%   prctile(pks,lo) and prctile(pks,hi) return the SAME number and the
%   half-open selection [tlo,thi) is empty. The narrowest band that can
%   resolve anything is roughly 100/u percentile points -- reported per panel
%   as `minband`, and in R.min_band.
%
%   Name-value options:
%     directions  cellstr of 'pos'/'neg'/'both'   (default {'pos','neg'})
%     marks       percentiles to draw as vertical lines (default [25 50 75])
%     logy        log-scale counts (default true)
%     xunits      'quantum' (default) or 'raw'
%     nbins       approximate bin count across the shared range (default 40)
%     fs, refPct, loFrac, hiFrac   forwarded to GC_WEIGHT_EVENTS
%     labels      channel names
%     printfig, printdir, printprefix, printres   optional .png per figure
%
%   R  struct array, one per direction, with:
%     .direction, .n_events (6x6), .n_unique (6x6), .min_band (6x6), .fig
%
%   Example:
%     R = gc_histogram(W);                       % pos + neg figures
%     R = gc_histogram(W, directions={'pos'}, marks=[49.5 50.5]);
%
%   See also GC_WEIGHT_EVENTS, GC_EVENT_GRID.

  arguments
    W
    opt.directions (1,:) cell = {'pos','neg'}
    opt.fs (1,1) double = 200
    opt.refPct (1,1) double = 100
    opt.loFrac (1,1) double = 0.2
    opt.hiFrac (1,1) double = 0.9
    opt.marks (1,:) double = [25 50 75]
    opt.logy (1,1) logical = true
    opt.xunits (1,:) char {mustBeMember(opt.xunits,{'quantum','raw'})} = 'quantum'
    opt.nbins (1,1) double = 40
    opt.xhi (1,1) double = 99.5       % clip x at this percentile of pks
    opt.labels (1,:) cell = {'SF','SE','EF','EE','BF','BE'}
    opt.printfig (1,1) logical = false
    opt.printdir (1,:) char = './'
    opt.printprefix (1,:) char = 'gc_hist'
    opt.printres (1,1) double = 300
    opt.verbose (1,1) logical = true
  end

  [nMN, nIa] = gc_wsize(W);
  R = struct('direction',{},'n_events',{},'n_unique',{},'min_band',{},'fig',{});

  for d = 1:numel(opt.directions)
    dir = opt.directions{d};
    nEv = zeros(nMN,nIa);  nUq = zeros(nMN,nIa);  mBand = nan(nMN,nIa);

    % ---- pass 1: collect events, derive the quantum and a shared x range --
    P = cell(nMN,nIa);
    for i = 1:nMN
      for j = 1:nIa
        [~, pk] = gc_weight_events(W, i, j, direction=dir, pct=100, ...
              fs=opt.fs, refPct=opt.refPct, ...
              loFrac=opt.loFrac, hiFrac=opt.hiFrac);
        P{i,j} = pk(:);
        nEv(i,j) = numel(pk);
      end
    end

    allp = vertcat(P{:});
    if isempty(allp)
      warning('gc_histogram:noEvents','No events for direction=%s.', dir);
      continue
    end
    % base quantum = finest spacing actually present in the pooled magnitudes
    du = diff(unique(allp));
    q  = min(du(du > 0));
    if isempty(q) || ~isfinite(q), q = 1; end

    xmaxq   = prctile(allp, opt.xhi) / q;        % shared upper limit, in quanta
    k       = max(1, round(xmaxq / opt.nbins));  % bin width, whole quanta
    edges_q = -0.5 + k*(0:ceil((xmaxq+k)/k));    % lattice points at bin centres

    if strcmp(opt.xunits,'quantum')
      sc = 1/q;  xlab_str = sprintf('|dw| peak  (quanta of %.4g)', q);
    else
      sc = 1;    xlab_str = '|dw| peak magnitude';
    end
    edges = edges_q * q * sc;
    xhi_v = xmaxq * q * sc;

    fig = figure('Color','w','Position',[120 90 1150 900]);
    tl = tiledlayout(nMN, nIa, 'TileSpacing','tight', 'Padding','compact');

    for i = 1:nMN
      for j = 1:nIa
        ax = nexttile;  hold(ax,'on');
        pks = P{i,j};

        if isempty(pks)
          text(ax, 0.5, 0.5, 'no events', 'Units','normalized', ...
               'HorizontalAlignment','center', 'FontSize',8, ...
               'Color',[0.5 0.5 0.5]);
          axis(ax,'off');
          continue
        end

        nUq(i,j)   = numel(unique(pks));
        mBand(i,j) = 100/nUq(i,j);

        histogram(ax, pks*sc, 'BinEdges', edges, ...
                  'FaceColor',[0.35 0.35 0.35], 'EdgeColor','none');

        % where percentile cuts would land
        for m = opt.marks
          xline(ax, prctile(pks, m)*sc, '-', 'Color',[0.85 0.1 0.1], ...
                'LineWidth',0.7, 'Alpha',0.8);
        end

        if opt.logy, set(ax,'YScale','log'); end
        xlim(ax, [0, xhi_v]);
        grid(ax,'on');  box(ax,'on');  ax.FontSize = 7;

        text(ax, 0.97, 0.93, sprintf('n=%d', nEv(i,j)), ...
             'Units','normalized','HorizontalAlignment','right','FontSize',7);
        text(ax, 0.97, 0.78, sprintf('u=%d', nUq(i,j)), ...
             'Units','normalized','HorizontalAlignment','right','FontSize',7, ...
             'Color',[0.1 0.3 0.7]);
        text(ax, 0.97, 0.63, sprintf('\\geq%.1f%%', mBand(i,j)), ...
             'Units','normalized','HorizontalAlignment','right','FontSize',7, ...
             'Color',[0.1 0.3 0.7], 'Interpreter','tex');

        if j == 1
          ylabel(ax, opt.labels{i}, 'FontWeight','bold', 'Rotation',0, ...
            'HorizontalAlignment','right', 'VerticalAlignment','middle');
        end
        if i == 1,   title(ax, opt.labels{j}, 'FontWeight','bold'); end
        if i ~= nMN, ax.XTickLabel = []; end
      end
    end

    xlabel(tl, sprintf('%s   (direction = %s)', xlab_str, dir));
    ylabel(tl, 'count');
    title(tl, sprintf(['Event magnitude distribution  (rows: MN,  cols: Ia)' ...
      '  --  n events, u distinct levels, \\geq narrowest usable pct band;' ...
      '  bin = %d quanta;  red = percentiles %s'], ...
      k, mat2str(opt.marks)), 'Interpreter','tex');

    fname = '';
    if opt.printfig
      if ~isfolder(opt.printdir), mkdir(opt.printdir); end
      fname = fullfile(opt.printdir, ...
                       sprintf('%s_%s.png', opt.printprefix, dir));
      exportgraphics(fig, fname, 'Resolution', opt.printres);
      if opt.verbose, fprintf('  saved %s\n', fname); end
    end

    if opt.verbose
      fprintf('direction=%-4s  events %d..%d   distinct levels %d..%d\n', ...
        dir, min(nEv(:)), max(nEv(:)), min(nUq(:)), max(nUq(:)));
      fprintf('  base quantum %.4g, bin width %d quanta\n', q, k);
      fprintf('  narrowest usable pct band across the grid: %.2f percentile points\n', ...
        max(mBand(:)));
    end

    R(d) = struct('direction',dir, 'n_events',nEv, 'n_unique',nUq, ...
                  'min_band',mBand, 'fig',fig);
  end
end
