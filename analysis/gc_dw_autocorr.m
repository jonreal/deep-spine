function [decorr_s, R] = gc_dw_autocorr(S, W, opt)
%GC_DW_AUTOCORR  Show that dw's autoregressive term is highly autocorrelated.
%   Plots a 6x6 grid (MN i x Ia j) of the normalized autocorrelation of the
%   weight derivative dw_ij vs lag in seconds. The slow, near-linear decay
%   (decorrelation time of order seconds, vs ~0 for white noise) demonstrates
%   that dw has long memory: its own past strongly predicts its present, so
%   the AR-history term in a Granger model cannot be treated as short/white.
%
%   Each panel marks the 1/e reference line and, where it occurs within the
%   window, the 1/e decorrelation time. Pruned/decaying connections are blank.
%
%   [decorr_s, R] = gc_dw_autocorr(S, W, ...)
%     decorr_s - nMN x nIa matrix of 1/e decorrelation times (s), NaN if the
%                ACF never reaches 1/e within maxLagSec (memory > window)
%     R        - struct with the ACF curves, lag axis, and options
%
%   Name-value options:
%     fs        - sampling rate, Hz (default 200)
%     maxLagSec - max one-sided lag shown, s (default 4; ~3 s decorrelation)
%     labels    - per-channel labels (default muscle labels)
  arguments
    S
    W
    opt.fs        (1,1) double = 200
    opt.maxLagSec (1,1) double = 4
    opt.labels    cell = {'SF','SE','EF','EE','BF','BE'}
  end

  dt     = 1/opt.fs;
  maxlag = round(opt.maxLagSec*opt.fs);
  thr    = exp(-1);

  nMN = size(S.act_MN,2);
  nIa = size(S.act_Ia,2);

  lag_s    = (0:maxlag)*dt;                 % one-sided lag axis (s)
  acf      = nan(nMN, nIa, maxlag+1);       % stored ACF curves
  decorr_s = nan(nMN, nIa);

  figure('Color','w','Position',[150 120 1000 900]);
  tl = tiledlayout(nMN, nIa, 'TileSpacing','tight', 'Padding','compact');
  xt = 0:1:floor(opt.maxLagSec);

  for i = 1:nMN
    for j = 1:nIa
      ax = nexttile;          % row-major fill matches i (outer), j (inner)
      hold(ax,'on');

      [~, ~, wij, dw] = gc_weight_events(W, i, j, fs=opt.fs);
      pruned = max(wij) <= 0 || wij(end) < 0.2*max(wij) || std(dw) == 0;

      if pruned
        set(ax,'Color',[0.95 0.95 0.95]);
        text(ax, opt.maxLagSec/2, 0.5, 'pruned', ...
          'HorizontalAlignment','center', 'Color',[0.7 0.7 0.7], 'FontSize',8);
      else
        [ac, lags] = xcorr(zscore(dw(:)), maxlag, 'coeff');
        a = ac(lags >= 0);
        acf(i,j,:) = a;
        plot(ax, lag_s, a, 'k');
        k = find(a < thr, 1, 'first');
        if ~isempty(k)
          decorr_s(i,j) = lag_s(k);
          plot(ax, lag_s(k), a(k), 'ro', 'MarkerFaceColor','r', 'MarkerSize',4);
        end
      end

      yline(ax, thr, ':', 'Color',[0.5 0.5 0.5]);   % 1/e reference
      ylim(ax,[0 1]); xlim(ax,[0 opt.maxLagSec]);
      xticks(ax, xt); grid(ax,'on'); box(ax,'on'); ax.FontSize = 8;

      % uniform outer-edge labels, driven by position (not pruning)
      if j == 1
        ylabel(ax, opt.labels{i}, 'FontWeight','bold', 'Rotation',0, ...
          'HorizontalAlignment','right', 'VerticalAlignment','middle');
      else
        ax.YTickLabel = [];
      end
      if i == 1,   title(ax, opt.labels{j}, 'FontWeight','bold'); end
      if i ~= nMN, ax.XTickLabel = []; end
    end
  end

  md = mean(decorr_s(:), 'omitnan');
  xlabel(tl, 'lag (s)');
  ylabel(tl, 'normalized ACF of dw');
  title(tl, sprintf(['dw autocorrelation  (rows: MN,  cols: Ia)  —  ' ...
    '1/e decorrelation \\approx %.2f s,  white noise \\approx 0'], md), ...
    'Interpreter','tex');

  R = struct('acf', acf, 'lag_s', lag_s, 'decorr_s', decorr_s, ...
    'fs', opt.fs, 'maxLagSec', opt.maxLagSec, 'labels', {opt.labels});
end
