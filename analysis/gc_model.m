
p = 5;
X1 = lagmatrix(Ia_i,0:p);
X1 = X1(p+1:end-p,:);

X2 = lagmatrix(kin,0:p);
X2 = X2(p+1:end-p,:);

% Lags if Y need to go in X1
Y = W_t(p+1:end-p,1);

tic
m0 = fitlm(X1,Y);
m0
toc

tic
m1 = fitlm([X1, X2],Y);
m1
toc

SSE0 = m0.SSE;  SSE1 = m1.SSE;
df0  = m0.DFE;  df1  = m1.DFE;

SST  = sum( (Y - mean(Y)).^2 );

F = ((SSE0 - SSE1)/(df0 - df1)) / (SSE1/df1);
p = 1 - fcdf(F, df0-df1, df1);

dR2        = (SSE0 - SSE1)/SST;    % == m1.R2 - m0.R2 (up to rounding)
partialR2  = (SSE0 - SSE1)/SSE0;

fprintf('F(%d,%d)=%.3f, p=%.3g\n', df0-df1, df1, F, p);
fprintf('R2: m0=%.4f, m1=%.4f, dR2=%.4f\n', ...
  m0.Rsquared.Ordinary, m1.Rsquared.Ordinary, dR2);
fprintf('partial R2=%.4f\n', partialR2);

