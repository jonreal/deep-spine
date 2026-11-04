function [dR2_MN,dR2_Ia,partialR2_MN,partialR2_Ia, ...
          dAIC_MN,dAIC_Ia,dBIC_MN,dBIC_Ia, ...
          F_MN,F_Ia,p_MN,p_Ia] = gc_run(S,W,i,j,opt)
  arguments
    S
    W
    i
    j
    opt.direction (1,:) char {mustBeMember(opt.direction,{'both','pos','neg'})} = 'both'
    opt.baseline (1,:) char {mustBeMember(opt.baseline,{'Ia','MN','both'})} = 'Ia'
    opt.detrend (1,:) char {mustBeMember(opt.detrend,{'none','diff','highpass'})} = 'none'
    opt.hpCutoff (1,1) double = 0.5
    opt.pct (1,1) double = 100
    opt.loFrac (1,1) double = 0.2
    opt.hiFrac (1,1) double = 0.9
    opt.refPct (1,1) double = 100
    opt.tWin   (1,2) double = [0 1]
    opt.pruneFrac (1,1) double = 0.2
    opt.nEvents (1,1) double = Inf
    opt.dWrule = []
    opt.nEventsMode (1,:) char = 'largest'
    opt.seed (1,1) double = 0
    opt.alpha = 0.025;
    opt.winSec = 1;
    opt.lag = 0.5;
    opt.feature = 'full';
  end


fs = 200;
dt = 1/fs;

% Which nested regressions to run. Reg 1 (MN baseline) costs the same as
% Reg 2 and is only a supporting result, so it is off by default.
doMN = any(strcmp(opt.baseline, {'MN','both'}));
doIa = any(strcmp(opt.baseline, {'Ia','both'}));

lag_win = round(opt.lag/dt);


[loc, pks, wij, dw, pruned] = gc_weight_events(W, i, j, pct=opt.pct, fs=fs, ...
                                       direction=opt.direction, ...
                                       loFrac=opt.loFrac, hiFrac=opt.hiFrac, ...
                                       refPct=opt.refPct, tWin=opt.tWin, ...
                                       pruneFrac=opt.pruneFrac, ...
                                       nEvents=opt.nEvents, ...
                                       dWrule=opt.dWrule, ...
                                       nEventsMode=opt.nEventsMode, ...
                                       seed=opt.seed);

% only analyze positive organization events for retained/growing weights

if pruned
    warning('Skipping pruned/decaying weight (%d,%d)', i,j);
    dR2_MN = NaN; dR2_Ia = NaN;
    partialR2_MN = NaN; partialR2_Ia = NaN;
    dAIC_MN = NaN; dAIC_Ia = NaN; dBIC_MN = NaN; dBIC_Ia = NaN;
    F_MN = NaN; F_Ia = NaN; p_MN = NaN; p_Ia = NaN;
    return
end

% Optional stationarity pre-processing of dw.
%
% Under feature='full' the baseline block is per-lag dw history, and dw is
% effectively a unit-root process (corr(dw(t),dw(t-1)) ~ 0.9999), so the
% baseline reaches R^2 ~ 1 and leaves almost no residual. Standard Granger
% causality requires stationarity; this is the step that supplies it.
%
% Applied AFTER event detection on purpose: events stay defined by the original
% weight dynamics, so 'none' vs 'diff'/'highpass' is a comparison over the same
% event set. Only the regression variable changes.
%
% Measured on example-1 (Aug 2026), feature='full', lag=0.2:
%   none      baseline R^2 = 1.0000
%   diff      baseline R^2 = 0.9947   <- actually removes the degeneracy
%   highpass  baseline R^2 = 0.9999   <- barely moves it at hpCutoff=0.5 Hz,
%                                        because the near-unit-root behaviour is
%                                        a lag-1 property at 200 Hz that a low
%                                        cutoff leaves intact
% Use 'diff' if you want the degeneracy genuinely gone. Its theoretical worry --
% that events are peaks of |dw| and the first difference is small at exactly
% those peaks -- did not bite in practice: F and p were stable across all three
% settings, which is why 'none' remains the default.
switch opt.detrend
  case 'none'
    % leave dw untouched; reproduces the pre-Aug-2026 behaviour exactly
  case 'diff'
    dw = [0, diff(dw(:).')];
  case 'highpass'
    [bhp, ahp] = butter(2, opt.hpCutoff/(fs/2), 'high');
    dw = filtfilt(bhp, ahp, dw(:).').';
    dw = dw(:).';
end

% drop events too close to the start to form a full pre-event history window
keep = loc > lag_win;
loc = loc(keep);
pks = pks(keep);

% A narrow tWin/gate can leave too few events to fit the unrestricted model;
% return NaN explicitly rather than letting fitlm throw.
if numel(loc) < 10
    warning('Too few events (%d) for (%d,%d); returning NaN', numel(loc), i, j);
    dR2_MN = NaN; dR2_Ia = NaN;
    partialR2_MN = NaN; partialR2_Ia = NaN;
    dAIC_MN = NaN; dAIC_Ia = NaN; dBIC_MN = NaN; dBIC_Ia = NaN;
    F_MN = NaN; F_Ia = NaN; p_MN = NaN; p_Ia = NaN;
    return
end

if strcmp(opt.feature,'full')
  % Pure lagged features, position AND velocity. Including velocity makes the
  % design rank deficient, but fitlm handles that correctly: it drops the
  % dependent columns and reports DFE = n - rank(X) rather than n - p - 1, so
  % the dfs feeding the nested F-test below stay valid. The "rank deficient to
  % within machine precision" warning is therefore benign, not something to
  % engineer around. Verified Aug 2026.
  lag_win = round(opt.lag/dt);
  Y_all      = nan(numel(loc), 1);
  Y_lags_all = nan(numel(loc), lag_win);
  X_MN_all   = nan(numel(loc), lag_win);
  X_Ia_all   = nan(numel(loc), lag_win);
  nKin = size(S.qpos,2) + size(S.qvel,2);
  X_kin_all = nan(numel(loc), lag_win*nKin);
  for k = 1:numel(loc)
    t = loc(k);
    hist_idx = t + (-lag_win:-1);
    Y_all(k,1)      = dw(t);
    Y_lags_all(k,:) = dw(hist_idx);
    X_MN_all(k,:) = S.act_MN(hist_idx,i)';
    X_Ia_all(k,:) = S.act_Ia(hist_idx,j)';
    X_kin_win = [S.qpos(hist_idx,:), S.qvel(hist_idx,:)];
    X_kin_all(k,:) = X_kin_win(:)';
  end


elseif strcmp(opt.feature,'mean')

  Y_mat = [];
  Y_all = [];
  Y_lags_all = [];
  X_kin_all = [];
  X_MN_all = [];
  X_Ia_all = [];
  for k = 1:numel(loc)
    t = loc(k);
    hist_idx = t + (-lag_win:-1);
    Y_all(k,1)      = dw(t);
    Y_lags_all(k,:) = mean(dw(hist_idx));
    X_MN_all(k,:) = mean(S.act_MN(hist_idx,i));
    X_Ia_all(k,:) = mean(S.act_Ia(hist_idx,j));
    X_kin_win = [S.qpos(hist_idx,:), S.qvel(hist_idx,:)];
    %X_kin_win = [S.qpos(hist_idx,:)];
    X_kin_all(k,:) = mean(X_kin_win,1);
  end


elseif strcmp(opt.feature,'bin')

  % Edit these based on xcorr. Example targets: 20--40 sample lead.
  % Three bins spanning the full lag window:
  % old, middle, recent
  edges = round([lag_win, 0.5*lag_win, 0.2*lag_win, 0]);
  bins = cell(1, numel(edges)-1);
  for b = 1:numel(edges)-1
    bins{b} = -edges(b) : -(edges(b+1)+1);
  end

  nBins = numel(bins);
  nKin  = size(S.qpos,2) + size(S.qvel,2);
  %nKin  = size(S.qpos,2);
  Y_all      = nan(numel(loc),1);
  Y_lags_all = nan(numel(loc),nBins);
  X_MN_all   = nan(numel(loc),nBins);
  X_Ia_all   = nan(numel(loc),nBins);
  X_kin_all  = nan(numel(loc),nBins*nKin);

  for k = 1:numel(loc)
    t = loc(k);
    Y_all(k,1) = dw(t);
    for b = 1:nBins
      idx_b = t + bins{b};
      Y_lags_all(k,b) = mean(dw(idx_b));
      X_MN_all(k,b)   = mean(S.act_MN(idx_b,i));
      X_Ia_all(k,b)   = mean(S.act_Ia(idx_b,j));
      X_kin_win = [S.qpos(idx_b,:), S.qvel(idx_b,:)];
      %X_kin_win = [S.qpos(idx_b,:)];
      X_kin_all(k,(b-1)*nKin + (1:nKin)) = mean(X_kin_win,1);
    end
  end

end

% S may be stored in single precision (see DS_LOAD_TRIAL). The regression
% features are small (n_events x ~10 columns), so cast to double here: the
% statistic of interest is the difference SSE0 - SSE1, and accumulating those
% sums of squares in single would round away part of it.
Y_all      = double(Y_all);
Y_lags_all = double(Y_lags_all);
X_MN_all   = double(X_MN_all);
X_Ia_all   = double(X_Ia_all);
X_kin_all  = double(X_kin_all);

fprintf('Y      %d x %d\n', size(Y_all));
fprintf('Ylags  %d x %d\n', size(Y_lags_all));
fprintf('MN     %d x %d\n', size(X_MN_all));
fprintf('kin    %d x %d\n', size(X_kin_all));
fprintf('Ia     %d x %d\n', size(X_Ia_all));

% Regression 1 -- baseline MN_i. Supporting result only (total kinematic
% dependence via the afferent path); Reg 2 below is the primary synergy
% test. It is the same size as Reg 2, so running it doubles the fitlm
% cost -- hence the default baseline='Ia'.
if doMN
  fprintf('-----Regression 1 - MN ----');
  tic
  m0 = fitlm([X_MN_all, Y_lags_all], Y_all);
  toc
  tic
  m1 = fitlm([X_MN_all, X_kin_all, Y_lags_all], Y_all);
  toc

  SSE0 = m0.SSE;  SSE1 = m1.SSE;
  df0  = m0.DFE;  df1  = m1.DFE;
  SST  = sum( (Y_all - mean(Y_all)).^2 );
  F = ((SSE0 - SSE1)/(df0 - df1)) / (SSE1/df1);
  p = 1 - fcdf(F, df0-df1, df1);

  dR2        = (SSE0 - SSE1)/SST;
  partialR2  = (SSE0 - SSE1)/SSE0;

  fprintf('F(%d,%d)=%.3f, p=%.3g\n', df0-df1, df1, F, p);
  fprintf('R2: m0=%.4f, m1=%.4f, dR2=%.4f\n', ...
    m0.Rsquared.Ordinary, m1.Rsquared.Ordinary, dR2);
  fprintf('partial R2=%.4f\n', partialR2);

  dR2_MN = dR2;
  partialR2_MN = partialR2;
  dAIC_MN = m1.ModelCriterion.AIC - m0.ModelCriterion.AIC;  % <0: kin earns its params
  dBIC_MN = m1.ModelCriterion.BIC - m0.ModelCriterion.BIC;
  F_MN = F;
  p_MN = p;
  fprintf('dAIC=%.2f, dBIC=%.2f\n', dAIC_MN, dBIC_MN);
else
  dR2_MN = NaN; partialR2_MN = NaN;
  dAIC_MN = NaN; dBIC_MN = NaN; F_MN = NaN; p_MN = NaN;
end

% Regression 2 -- baseline Ia_j. THE primary test: Ia_j is muscle-specific
% f(len_j,vel_j), so kinematics beyond it is the target muscle's own
% movement and cross-joint coupling.
if doIa
  fprintf('\n\n-----Regression 2 - Ia ----');
  tic
  m0 = fitlm([X_Ia_all, Y_lags_all], Y_all);
  toc
  tic
  m1 = fitlm([X_Ia_all, X_kin_all, Y_lags_all], Y_all);
  toc

  SSE0 = m0.SSE;  SSE1 = m1.SSE;
  df0  = m0.DFE;  df1  = m1.DFE;
  SST  = sum( (Y_all - mean(Y_all)).^2 );
  F = ((SSE0 - SSE1)/(df0 - df1)) / (SSE1/df1);
  p = 1 - fcdf(F, df0-df1, df1);

  dR2        = (SSE0 - SSE1)/SST;
  partialR2  = (SSE0 - SSE1)/SSE0;

  fprintf('F(%d,%d)=%.3f, p=%.3g\n', df0-df1, df1, F, p);
  fprintf('R2: m0=%.4f, m1=%.4f, dR2=%.4f\n', ...
    m0.Rsquared.Ordinary, m1.Rsquared.Ordinary, dR2);
  fprintf('partial R2=%.4f\n', partialR2);

  dR2_Ia = dR2;
  partialR2_Ia = partialR2;
  dAIC_Ia = m1.ModelCriterion.AIC - m0.ModelCriterion.AIC;  % <0: kin earns its params
  dBIC_Ia = m1.ModelCriterion.BIC - m0.ModelCriterion.BIC;
  F_Ia = F;
  p_Ia = p;
  fprintf('dAIC=%.2f, dBIC=%.2f\n', dAIC_Ia, dBIC_Ia);
else
  dR2_Ia = NaN; partialR2_Ia = NaN;
  dAIC_Ia = NaN; dBIC_Ia = NaN; F_Ia = NaN; p_Ia = NaN;
end

