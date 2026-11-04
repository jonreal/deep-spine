load(['../experiments/hebbian/logs/legacy/',...
    'exp9/trials/1756957634/trial_data.mat']);




%Y1 = [S.act_MN(:,1), S.act_MN(:,2), S.act_MN(:,3), ...
%      S.act_MN(:,4), S.act_MN(:,5), S.act_MN(:,6)];
%Y2 = [wfun(1,1)', wfun(2,2)', wfun(3,3)', wfun(4,4)', wfun(5,5)', wfun(6,6)'];
%
%
%%[h,pvalue,stat,cvalue] = gctest(Y1,Y2)
%
%
%Y1 = [S.act_Ia(:,1), S.act_Ia(:,2), S.act_Ia(:,3), ...
%      S.act_Ia(:,4), S.act_Ia(:,5), S.act_Ia(:,6)];
%Y2 = [wfun(1,1)', wfun(2,2)', wfun(3,3)', wfun(4,4)', wfun(5,5)', wfun(6,6)'];
%
%
%%[h,pvalue,stat,cvalue] = gctest(Y1,Y2)
%
%results = [];
%for i=1:6
%  [h,pvalue,stat,cvalue] = gctest(Y1(:,i),Y2(:,1),Y1(:,);
%  results = [results; [h, pvalue]];
%end
%Res


%% Setup
wfun = @(i,j) arrayfun(@(t) W{t}(i,j), 1:numel(W))';
nMuscles = 6;

ds = 1;

idx = 1:ds:numel(W);
kin = [S.qpos(idx,:), S.qvel(idx,:)];
Ia_i = S.act_Ia(idx, :);

nsteps = size(kin,1);


max_lag = 20;

W_t = zeros(nsteps,36);
C_w = zeros(36,3);
L_t = zeros(36,2);
cnt = 1;
for i=1:nMuscles
  for j=1:nMuscles

    w_ij = wfun(i,j);
    W_t(:,cnt) = w_ij;
    L_t(cnt,:) = [i, j];

    if i==j
      if (i==1) | (i==2)
        C_w(cnt,:) = [0.6 0.6 0];
      elseif (i==3) | (i==4)
        C_w(cnt,:) = [0.4 0.4 0];
      elseif (i==5) | (i==6)
        C_w(cnt,:) = [0.2 0.2 0];
      end
    elseif mod(i+j,2) == 0
      if (i==1) | (i==2)
        C_w(cnt,:) = [0 0.6 0];
      elseif (i==3) | (i==4)
        C_w(cnt,:) = [0 0.4 0];
      elseif (i==5) | (i==6)
        C_w(cnt,:) = [0 0.2 0];
      end
    else
      if (i==1) | (i==2)
        C_w(cnt,:) = [0 0 0.6];
      elseif (i==3) | (i==4)
        C_w(cnt,:) = [0 0 0.4];
      elseif (i==5) | (i==6)
        C_w(cnt,:) = [0 0 0.2];
      end

    end


    %str = ['(',num2str(i),', ',num2str(j), ')\n','\tCalculating lag '];
    %fprintf(str);
    %tic
    %[opt_lag, bic_all] = select_gc_lag_fast(w_ij,Ia_i,20);
    %fprintf(', lag = %i\n',opt_lag);
    %toc
    %tic
    %str = ['Calculating grainger-causality. (', ...
    %      num2str(i),', ',num2str(j), ')\n'];
    %fprintf(str);
    %[h_kin, p_kin, stat_kin] = gctest(w_ij, kin, Ia_i,'NumLags',opt_lag);
    %toc
    %R(:,cnt) = [h_kin, p_kin, stat_kin];
    cnt = cnt + 1;
  end
end
D = {kin,Ia_i};

return

% Test: Kin → w | neural
% (does kinematics Granger-cause weight, conditioned on neural?)

    % Test: Neural → w | kin
   % [h_neu, p_neu, stat_neu] = gctest(neural, w_ij, kin,'NumLags',4);



[b,a] = butter(1,0.05,'low');

%% Test
results = [];

for i = 1:nMuscles
  for j = 1:nMuscles
    fprintf('Testing w(%d,%d)...\n', i, j);

    w_ij = wfun(i,j); w_ij = w_ij(idx);
    Ia_i = S.act_Ia(idx, i);
    MN_j = S.act_MN(idx, j);
    neural = [Ia_i, MN_j];

    tic
    Y1 = filtfilt(b,a,w_ij);
    Y2 = filtfilt(b,a,kin);
    Y3 = filtfilt(b,a,Ia_i);

    figure;
      subplot(211);
      plot(Ia_i,'k');
      subplot(212);
      plot(Y1,'r');

    [h_kin, p_kin, stat_kin] = gctest(w_ij, kin, Ia_i,'NumLags',4)
    toc


    % Test: Kin → w | neural
    % (does kinematics Granger-cause weight, conditioned on neural?)
    %[h_kin, p_kin, stat_kin] = gctest(kin, w_ij, neural,'NumLags',4);

    % Test: Neural → w | kin
   % [h_neu, p_neu, stat_neu] = gctest(neural, w_ij, kin,'NumLags',4);

    isHomo = (i == j);
    results = [results; i, j, isHomo, p_neu, stat_neu, p_kin, stat_kin]
  end
end


%% Partition results
T = array2table(results, 'VariableNames', ...
    {'i', 'j', 'isHomo', 'p_neu', 'stat_neu', 'p_kin', 'stat_kin'});

% Add category
T.category = repmat("", height(T), 1);
for k = 1:height(T)
    i = T.i(k); j = T.j(k);
    if i == j
        T.category(k) = "homo";
    elseif mod(i+j, 2) == 0
        T.category(k) = "synergy";
    else
        T.category(k) = "anti";
    end
end

%% Summary by category
cats = ["homo", "synergy", "anti"];
fprintf('\n=== Kin → w | neural ===\n');
for c = cats
    subset = T(T.category == c, :);
    fprintf('%s (mean stat): %.4f\n', c, mean(subset.stat_kin));
end

fprintf('\n=== Neural → w | kin ===\n');
for c = cats
    subset = T(T.category == c, :);
    fprintf('%s (mean stat): %.4f\n', c, mean(subset.stat_neu));
end


W_final = W{end};
for c = cats
    subset = T(T.category == c, :);
    weights = arrayfun(@(k) W_final(T.i(k), T.j(k)), find(T.category == c));
    fprintf('%s (mean final weight): %.4f\n', c, mean(weights));
end


function [opt_lag, bic_all] = select_gc_lag_fast(y, z_cond, max_lag)
% Fast BIC lag selection using pre-built lag matrix + incremental QR
%
%   y       : (T x 1)
%   z_cond  : (T x m)
%   max_lag : scalar

[T, m] = size(z_cond);
n_cols_per_lag = 1 + m;  % one lag of y + one lag of z_cond

% Pre-build the full lag matrix for max_lag all at once
%   Columns: [y(t-1) z(t-1) | y(t-2) z(t-2) | ... | y(t-p) z(t-p)]
n_obs = T - max_lag;
y_dep = y(max_lag+1 : T);

X_full = ones(n_obs, 1 + max_lag * n_cols_per_lag);  % intercept + all lags
col = 2;
for k = 1:max_lag
    rows = max_lag+1-k : T-k;
    X_full(:, col)              = y(rows);
    X_full(:, col+1 : col+m)   = z_cond(rows, :);
    col = col + n_cols_per_lag;
end

% Now evaluate BIC at each lag p by using first 1 + p*n_cols_per_lag columns
bic_all = nan(max_lag, 1);
for p = 1:max_lag
    nc = 1 + p * n_cols_per_lag;
    Xp = X_full(:, 1:nc);

    % QR is faster than backslash for repeated similar-size problems
    [Q, R] = qr(Xp, 0);
    resid = y_dep - Q * (Q' * y_dep);
    sigma2 = (resid' * resid) / n_obs;

    bic_all(p) = n_obs * log(sigma2) + nc * log(n_obs);
end

[~, opt_lag] = min(bic_all);
end
