



NumLags = 200;

figure;

ncol = 36;
nrow = 7;
xlab = 0.02;
ylab = 0.02;
ysp = 0.005;
xsp = 0.005;
w = (1 - 2*ylab - (ncol-1)*xsp)/ncol;
h = (1 - 2*xlab - (nrow-1)*ysp)/nrow;

Ia_labels = {'Ia_{SF}','Ia_{SE}','Ia_{EF}','Ia_{EE}','Ia_{BF}','Ia_{BE}'};

xs = ylab;
ys = 1-xlab-h;

L = zeros(36,6);
for k=1:36
  y = gradient(W_t(:,k));
  for i=1:6
    x = Ia_i(:,i);
    id = L_t(k,:);

    [xcf,lags] = crosscorr(y-mean(y),x-mean(x),NumLags=NumLags);
    [~,idx] = max(xcf);
    L(k,i) = lags(idx);

    pos = [xs,ys,w,h];
    ax = axes('Position',pos);
    plot(lags,xcf,'k','linewidth',0.1); hold on;
    scatter(lags(idx),xcf(idx),10,'filled');
    plot([lags(idx),lags(idx)],[-xcf(idx),xcf(idx)],'--r');
    axis off;
    ax.XAxis.Visible = 'on';
    xticks([lags(idx)]);

    if i==1
      pos_an = pos;
      pos_an(1) = pos_an(1)-2*xsp;
      an = annotation('textbox','position',pos_an);
      an.String = ['W_{',num2str(id(1)),num2str(id(2)),'}'];
      an.EdgeColor = 'none';
      an.FontSize = 14;
    end
    if k == 1
      pos_an = pos;
      pos_an(1) = pos_an(1)-2*xsp;
      an = annotation('textbox','position',pos_an);
      an.String = Ia_labels{i};
      an.HorizontalAlignment = 'left';
      an.VerticalAlignment = 'middle';
      an.EdgeColor = 'none';
      an.FontSize = 14;
    end

    ys = ys - h - ysp;


  end
  xs = xs + w + xsp;
  ys = 1-xlab-h;
end

xs = ylab;
ys = ys - (nrow-1)*h - (nrow-2)*ysp - 2*ysp;
w = ncol*w + (ncol-1)*xsp;
pos = [xs,ys,w,h-2*ysp];
ax = axes('Position',pos);
b = bar(mean(L,2),'k');
ax.Color = 'none';
xlim([1-b.BarWidth/2, 36+b.BarWidth/2]);
ax.XAxis.Visible = 'off';

figure;
h = histogram(mean(L,2));
ax=gca;
ax.Color = 'none';






return

ann = annotation('textbox','position',[0.1, 0.1, 0.8, 0.9]);
ann.String = ['\Delta W_{', num2str(L_t(1,1)), num2str(L_t(1,2)),'}'];


return


m1 = fitlm(Ia_i, W_t(:,1));

return








%load(['../../experiments/hebbian/logs/',...
%    'exp9/trials/1756957634/trial_data.mat']);
%%Y1 = [S.act_MN(:,1), S.act_MN(:,2), S.act_MN(:,3), ...
%%      S.act_MN(:,4), S.act_MN(:,5), S.act_MN(:,6)];
%%Y2 = [wfun(1,1)', wfun(2,2)', wfun(3,3)', wfun(4,4)', wfun(5,5)', wfun(6,6)'];
%%
%%
%%%[h,pvalue,stat,cvalue] = gctest(Y1,Y2)
%%
%%
%%Y1 = [S.act_Ia(:,1), S.act_Ia(:,2), S.act_Ia(:,3), ...
%%      S.act_Ia(:,4), S.act_Ia(:,5), S.act_Ia(:,6)];
%%Y2 = [wfun(1,1)', wfun(2,2)', wfun(3,3)', wfun(4,4)', wfun(5,5)', wfun(6,6)'];
%%
%%
%%%[h,pvalue,stat,cvalue] = gctest(Y1,Y2)
%%
%%results = [];
%%for i=1:6
%%  [h,pvalue,stat,cvalue] = gctest(Y1(:,i),Y2(:,1),Y1(:,);
%%  results = [results; [h, pvalue]];
%%end
%%Res
%
%
%%% Setup
%wfun = @(i,j) arrayfun(@(t) W{t}(i,j), 1:numel(W))';
%nMuscles = 6;
%
%ds = 1;
%
%idx = 1:ds:numel(W);
%kin = [S.qpos(idx,:), S.qvel(idx,:)];
%Ia_i = S.act_Ia(idx, :);
%
%nsteps = size(kin,1);
%
%
%max_lag = 20;
%
%W_t = zeros(nsteps,36);
%C_w = zeros(36,3);
%cnt = 1;
%for i=1:nMuscles
%  for j=1:nMuscles
%
%    w_ij = wfun(i,j);
%    W_t(:,cnt) = w_ij;
%
%    if i==j
%      if (i==1) | (i==2)
%        C_w(cnt,:) = [0.6 0.6 0];
%      elseif (i==3) | (i==4)
%        C_w(cnt,:) = [0.4 0.4 0];
%      elseif (i==5) | (i==6)
%        C_w(cnt,:) = [0.2 0.2 0];
%      end
%    elseif mod(i+j,2) == 0
%      if (i==1) | (i==2)
%        C_w(cnt,:) = [0 0.6 0];
%      elseif (i==3) | (i==4)
%        C_w(cnt,:) = [0 0.4 0];
%      elseif (i==5) | (i==6)
%        C_w(cnt,:) = [0 0.2 0];
%      end
%    else
%      if (i==1) | (i==2)
%        C_w(cnt,:) = [0 0 0.6];
%      elseif (i==3) | (i==4)
%        C_w(cnt,:) = [0 0 0.4];
%      elseif (i==5) | (i==6)
%        C_w(cnt,:) = [0 0 0.2];
%      end
%
%    end
%
%
%    %str = ['(',num2str(i),', ',num2str(j), ')\n','\tCalculating lag '];
%    %fprintf(str);
%    %tic
%    %[opt_lag, bic_all] = select_gc_lag_fast(w_ij,Ia_i,20);
%    %fprintf(', lag = %i\n',opt_lag);
%    %toc
%    %tic
%    %str = ['Calculating grainger-causality. (', ...
%    %      num2str(i),', ',num2str(j), ')\n'];
%    %fprintf(str);
%    %[h_kin, p_kin, stat_kin] = gctest(w_ij, kin, Ia_i,'NumLags',opt_lag);
%    %toc
%    %R(:,cnt) = [h_kin, p_kin, stat_kin];
%    cnt = cnt + 1;
%  end
%end
%D = {kin,Ia_i};
%
%
%% Test: Kin → w | neural
%% (does kinematics Granger-cause weight, conditioned on neural?)
%
%    % Test: Neural → w | kin
%   % [h_neu, p_neu, stat_neu] = gctest(neural, w_ij, kin,'NumLags',4);
%
%
%
%return
%
%  return
%
%
%[b,a] = butter(1,0.05,'low');
%
%%% Test
%results = [];
%
%for i = 1:nMuscles
%  for j = 1:nMuscles
%    fprintf('Testing w(%d,%d)...\n', i, j);
%
%    w_ij = wfun(i,j); w_ij = w_ij(idx);
%    Ia_i = S.act_Ia(idx, i);
%    MN_j = S.act_MN(idx, j);
%    neural = [Ia_i, MN_j];
%
%    tic
%    Y1 = filtfilt(b,a,w_ij);
%    Y2 = filtfilt(b,a,kin);
%    Y3 = filtfilt(b,a,Ia_i);
%
%    figure;
%      subplot(211);
%      plot(Ia_i,'k');
%      subplot(212);
%      plot(Y1,'r');
%
%    pause
%    [h_kin, p_kin, stat_kin] = gctest(w_ij, kin, Ia_i,'NumLags',4)
%    toc
%
%    pause
%
%
%    % Test: Kin → w | neural
%    % (does kinematics Granger-cause weight, conditioned on neural?)
%    %[h_kin, p_kin, stat_kin] = gctest(kin, w_ij, neural,'NumLags',4);
%
%    % Test: Neural → w | kin
%   % [h_neu, p_neu, stat_neu] = gctest(neural, w_ij, kin,'NumLags',4);
%
%    isHomo = (i == j);
%    results = [results; i, j, isHomo, p_neu, stat_neu, p_kin, stat_kin]
%  end
%end
%
%
%%% Partition results
%T = array2table(results, 'VariableNames', ...
%    {'i', 'j', 'isHomo', 'p_neu', 'stat_neu', 'p_kin', 'stat_kin'});
%
%% Add category
%T.category = repmat("", height(T), 1);
%for k = 1:height(T)
%    i = T.i(k); j = T.j(k);
%    if i == j
%        T.category(k) = "homo";
%    elseif mod(i+j, 2) == 0
%        T.category(k) = "synergy";
%    else
%        T.category(k) = "anti";
%    end
%end
%
%%% Summary by category
%cats = ["homo", "synergy", "anti"];
%fprintf('\n=== Kin → w | neural ===\n');
%for c = cats
%    subset = T(T.category == c, :);
%    fprintf('%s (mean stat): %.4f\n', c, mean(subset.stat_kin));
%end
%
%fprintf('\n=== Neural → w | kin ===\n');
%for c = cats
%    subset = T(T.category == c, :);
%    fprintf('%s (mean stat): %.4f\n', c, mean(subset.stat_neu));
%end
%
%
%W_final = W{end};
%for c = cats
%    subset = T(T.category == c, :);
%    weights = arrayfun(@(k) W_final(T.i(k), T.j(k)), find(T.category == c));
%    fprintf('%s (mean final weight): %.4f\n', c, mean(weights));
%end
%
%
%function [opt_lag, bic_all] = select_gc_lag_fast(y, z_cond, max_lag)
%% Fast BIC lag selection using pre-built lag matrix + incremental QR
%%
%%   y       : (T x 1)
%%   z_cond  : (T x m)
%%   max_lag : scalar
%
%[T, m] = size(z_cond);
%n_cols_per_lag = 1 + m;  % one lag of y + one lag of z_cond
%
%% Pre-build the full lag matrix for max_lag all at once
%%   Columns: [y(t-1) z(t-1) | y(t-2) z(t-2) | ... | y(t-p) z(t-p)]
%n_obs = T - max_lag;
%y_dep = y(max_lag+1 : T);
%
%X_full = ones(n_obs, 1 + max_lag * n_cols_per_lag);  % intercept + all lags
%col = 2;
%for k = 1:max_lag
%    rows = max_lag+1-k : T-k;
%    X_full(:, col)              = y(rows);
%    X_full(:, col+1 : col+m)   = z_cond(rows, :);
%    col = col + n_cols_per_lag;
%end
%
%% Now evaluate BIC at each lag p by using first 1 + p*n_cols_per_lag columns
%bic_all = nan(max_lag, 1);
%for p = 1:max_lag
%    nc = 1 + p * n_cols_per_lag;
%    Xp = X_full(:, 1:nc);
%
%    % QR is faster than backslash for repeated similar-size problems
%    [Q, R] = qr(Xp, 0);
%    resid = y_dep - Q * (Q' * y_dep);
%    sigma2 = (resid' * resid) / n_obs;
%
%    bic_all(p) = n_obs * log(sigma2) + nc * log(n_obs);
%end
%
%[~, opt_lag] = min(bic_all);
%end
