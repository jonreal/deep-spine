function [T, R] = gc_lag_estimation(S, W, opt)
%GC_LAG_ESTIMATION  Estimate predictor->dw lags for Granger-style analysis.
%   For each (MN i, Ia j) connection, cross-correlates the weight derivative
%   dw_ij against its candidate predictors on the full time series and reports
%   the lag of peak coupling:
%     x_MN_i -> dw_ij   predictive delay of the motoneuron
%     x_Ia_j -> dw_ij   predictive delay of the Ia afferent
%
%   Correlations use xcorr(dw, x), so POSITIVE lags mean the predictor x leads
%   dw (the causal/predictive side); only that side is searched. The peak is
%   taken by |corr| (so inhibitory couplings register) but the SIGNED peak
%   correlation is returned. The peak lag is robust to the session-long drift
%   in dw: it is a local bump above the slow background, not the background
%   itself, so no detrending is applied.
%
%   Use these lags to set the history window in the GC regression (e.g. a lag
%   window spanning ~0 .. max peak lag across predictors).
%
%   [T, R] = gc_lag_estimation(S, W, ...)
%     T - long-format table, one row per connection (lags in seconds)
%     R - struct of nMN x nIa matrices + options for downstream use
%
%   Name-value options:
%     fs        - sampling rate, Hz (default 200)
%     maxLagSec - max lag searched, s (default 0.5)
%     labels    - per-channel labels (default muscle labels)
%     dWrule    - rule-computed per-step dW (ds_load_dw_rule), passed to
%                 GC_WEIGHT_EVENTS; default [] = gradient of W
  arguments
    S
    W
    opt.fs        (1,1) double = 200
    opt.maxLagSec (1,1) double = 0.5
    opt.labels    cell = {'SF','SE','EF','EE','BF','BE'}
    opt.dWrule = []
  end

  dt     = 1/opt.fs;
  maxlag = round(opt.maxLagSec*opt.fs);

  nMN = size(S.act_MN,2);
  nIa = size(S.act_Ia,2);

  % zscore predictors once (mean-removed; xcorr 'coeff' handles scale)
  XMN = zscore(S.act_MN);
  XIa = zscore(S.act_Ia);

  mn_peaklag = nan(nMN,nIa);  mn_peakcor = nan(nMN,nIa);
  ia_peaklag = nan(nMN,nIa);  ia_peakcor = nan(nMN,nIa);

  for i = 1:nMN
    for j = 1:nIa
      % Canonical weight trace + derivative (peaks discarded here)
      [~, ~, wij, dw] = gc_weight_events(W, i, j, fs=opt.fs, dWrule=opt.dWrule);

      % Skip pruned/decaying or flat weights (no organization to explain)
      if max(wij) <= 0 || wij(end) < 0.2*max(wij) || std(dw) == 0
        continue
      end

      dwz = zscore(dw(:));
      [mn_peaklag(i,j), mn_peakcor(i,j)] = xcorr_lag(dwz, XMN(:,i), maxlag, dt);
      [ia_peaklag(i,j), ia_peakcor(i,j)] = xcorr_lag(dwz, XIa(:,j), maxlag, dt);
    end
  end

  % ---- pack struct ----
  R = struct( ...
    'mn_peaklag', mn_peaklag, 'mn_peakcor', mn_peakcor, ...
    'ia_peaklag', ia_peaklag, 'ia_peakcor', ia_peakcor, ...
    'fs', opt.fs, 'maxLagSec', opt.maxLagSec, 'labels', {opt.labels});

  % ---- long-format table ----
  labels = opt.labels;
  if numel(labels) < max(nMN,nIa)
    labels = compose('%d', 1:max(nMN,nIa));   % numeric fallback
  end
  [I,J] = ndgrid(1:nMN, 1:nIa);
  conn  = strings(numel(I),1);
  for n = 1:numel(I)
    conn(n) = labels{I(n)} + "←" + labels{J(n)};
  end
  T = table(I(:), J(:), conn, ...
      mn_peaklag(:), mn_peakcor(:), ia_peaklag(:), ia_peakcor(:), ...
      'VariableNames', {'MN','Ia','Connection', ...
        'mn_peaklag_s','mn_peakcorr','ia_peaklag_s','ia_peakcorr'});

  % ---- figure: 1 row x 2 cols, shared color scale (both are peak lags) ----
  figure('Color','w','Position',[200 300 760 360]);
  hi = max([mn_peaklag(:); ia_peaklag(:)], [], 'omitnan');
  if ~(hi > 0), hi = opt.maxLagSec; end
  panels = {mn_peaklag, 'peak lag: x_{MN} \rightarrow dw'; ...
            ia_peaklag, 'peak lag: x_{Ia} \rightarrow dw'};
  for p = 1:size(panels,1)
    M = panels{p,1};
    M(isnan(M)) = 0;                      % pruned cells as zero (cleaner)
    subplot(1,2,p);
    imagesc(M);
    axis image; clim([0 hi]);
    cb = colorbar; cb.Label.String = 'lag (s)';
    xlabel('Ia index j'); ylabel('MN index i');
    xticks(1:nIa); yticks(1:nMN);
    if numel(opt.labels) >= max(nMN,nIa)
      xticklabels(opt.labels); yticklabels(opt.labels);
    end
    title(panels{p,2});
  end
  sgtitle(sprintf('Predictor \\rightarrow dw peak lag (full series), fs=%g Hz, maxLag=%.3g s', ...
    opt.fs, opt.maxLagSec));
end


function [peaklag_s, peakcorr] = xcorr_lag(yz, x, maxlag, dt)
% xcorr(dw, x): positive lag = x leads dw (predictive). Peak by |corr|;
% returns the signed peak correlation and its lag in seconds.
  [c, lags] = xcorr(yz, zscore(x(:)), maxlag, 'coeff');
  pos = lags >= 0;
  cp  = c(pos);
  lp  = lags(pos);

  [~, mi]   = max(abs(cp));
  peakcorr  = cp(mi);
  peaklag_s = lp(mi)*dt;
end
