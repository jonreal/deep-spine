function ds_plot_motif_gui(tab, opt)
%DS_PLOT_MOTIF_GUI  Clickable version of DS_PLOT_MOTIF_HEATMAPS: click a cell, open its dashboard.
%
%   Draws the usual motif heatmap grid by calling DS_PLOT_MOTIF_HEATMAPS (which is left
%   untouched), then makes every panel clickable. Clicking near a swept point
%   opens that simulation's dashboard PNG in the system viewer.
%
%   ds_plot_motif_gui(summary_table)
%   ds_plot_motif_gui(summary_table, root='<repo>/logs/hebbian-sweep/SR0-MR0')
%
%   The table needs a 'root' column (added by DS_PROCESS_BABBLE) so that a
%   POOLED table spanning several sweep trees can still find each run. Older
%   tables lack it -- pass root= as a fallback, which then applies to all rows.
%
%   Dashboards only exist if ds_process_babble ran with printfig=true. If the
%   PNG is missing the click prints the sim-ID and the path it looked for,
%   rather than silently doing nothing.
%
%   See also DS_PLOT_MOTIF_HEATMAPS, DS_PROCESS_BABBLE.

  arguments
    tab
    opt.root (1,:) char = ''      % fallback when the table has no 'root' column
    opt.cmap = turbo(1024)
    opt.selzl (1,2) double = [0 0.30]
    opt.dotsize (1,1) double = 20   % config dots are size 5 by default,
                                    % too small to hit reliably
  end

  ds_plot_motif_heatmaps(tab, cmap=opt.cmap, selzl=opt.selzl);
  fig = gcf;

  % Heatmap panels are the axes containing a surf; the colourbars hold images,
  % so this picks the grid out without depending on child ordering.
  axs = findobj(fig, 'Type', 'axes');
  isHM = arrayfun(@(a) ~isempty(findobj(a,'Type','surface')), axs);
  hm = axs(isHM);
  if isempty(hm), warning('ds_plot_motif_gui:noPanels','No heatmap axes found.'); return; end

  % sort into row-major order: top row first, then left to right
  P = cell2mat(arrayfun(@(a) a.Position(1:2), hm, 'UniformOutput', false));
  [~,ord] = sortrows([-P(:,2), P(:,1)]);
  hm = hm(ord);

  gains = unique(tab.('heb_kwa_gai'));
  ncols = 4;

  % Annotations live on the scribe layer ABOVE every axes, and ds_plot_motif_heatmaps's
  % column titles are positioned to overlap ~96% of each top-row panel. Left
  % pickable they swallow every click before it reaches the data.
  set(findall(fig,'Type','textboxshape'), 'HitTest','off', 'PickableParts','none');

  nDots = 0;
  for k = 1:numel(hm)
    ax = hm(k);
    g  = gains(min(ceil(k/ncols), numel(gains)));

    % The surf must not intercept -- the clickable thing is the scatter of
    % config points drawn on top of it.
    set(findobj(ax,'Type','surface'), 'HitTest','off', 'PickableParts','none');

    sc = findobj(ax, 'Type', 'scatter');
    for q = 1:numel(sc)
      sc(q).HitTest       = 'on';
      sc(q).PickableParts = 'visible';
      sc(q).SizeData      = opt.dotsize;
      sc(q).ButtonDownFcn = @(src,evt) on_click(src, evt, tab, g, opt.root);
      nDots = nDots + numel(sc(q).XData);
    end
  end

  fprintf(['ds_plot_motif_gui: click a black config dot to open that run''s dashboard.\n' ...
           '  %d panels, %d dots, gains %s\n'], numel(hm), nDots, mat2str(gains'));
end

% ---------------------------------------------------------------------------
function on_click(src, evt, tab, gain, rootFallback)
  % scatter clicks carry the 3-D intersection directly; fall back to the
  % parent axes' CurrentPoint if that is unavailable
  if ~isempty(evt) && isprop(evt,'IntersectionPoint') && numel(evt.IntersectionPoint) >= 2
    cp = evt.IntersectionPoint(1:2);
  else
    ax = ancestor(src,'axes');
    cp = ax.CurrentPoint(1,1:2);
  end
  sel = tab.('heb_kwa_gai') == gain;
  T = tab(sel,:);
  if isempty(T), fprintf('no rows at kappa = %g\n', gain); return; end

  pb  = T.sup_kwa_bab_max_pro;
  str = T.opt_max_mus_str;

  % nearest swept point. ds_plot_motif_heatmaps plots x = log10(p_b) on a linear axis and
  % y = strength, so cp is already in (log10(p_b), strength); normalise both
  % axes or the wider range dominates the distance.
  x = log10(pb);
  lx = range(x);    if lx == 0, lx = 1; end
  ly = range(str);  if ly == 0, ly = 1; end
  d = ((x - cp(1))/lx).^2 + ((str - cp(2))/ly).^2;
  [~,k] = min(d);

  simid = char(string(T{k,'sim-ID'}));
  if ismember('root', T.Properties.VariableNames)
    root = char(string(T{k,'root'}));
  else
    root = rootFallback;
  end

  fprintf('\nsim-ID %s   kappa=%g  lambda=%g  p_b=%.4f\n', simid, gain, str(k), pb(k));
  for f = {'r_mean','s_mean','a_mean','sel_mean'}
    if ismember(f{1}, T.Properties.VariableNames)
      fprintf('   %-9s %.4f\n', f{1}, T{k,f{1}});
    end
  end

  if isempty(root)
    fprintf(['   no root: table has no ''root'' column and none was passed.\n' ...
             '   re-run ds_process_babble, or call with root=<sweep dir>.\n']);
    return
  end

  dash = fullfile(root, 'analysis', 'dashboard');
  hits = dir(fullfile(dash, [simid '*.png']));
  if isempty(hits)
    fprintf(['   no dashboard for sim-ID %s\n' ...
             '   looked in: %s\n' ...
             '   re-run ds_process_babble with printfig=true to generate it.\n'], ...
             simid, dash);
    return
  end

  png = fullfile(hits(1).folder, hits(1).name);
  fprintf('   opening %s\n', hits(1).name);
  % quote the path: these filenames contain [ ] and , which the shell would eat
  if ispc
    winopen(png);
  elseif ismac
    system(['open "' png '"']);
  else
    system(['xdg-open "' png '" &']);
  end
end
