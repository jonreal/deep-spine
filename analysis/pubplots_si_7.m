% PUBPLOTS_SI_7  SI figure: Ia afferent -> weight-change peak lag per connection.
%
% Script -- expects the high-rate trial in the workspace:
%   [S,W,P] = ds_load_trial('../experiments/hebbian/logs/example-3/1790298957');
%   dW = ds_load_dw_rule('../experiments/hebbian/logs/example-3/1790298957');
%   pubplots_si_7
%
% Lags come from gc_lag_estimation (unchanged method): for each connection
% (MN i <- Ia j), the cross-correlation of the weight change dw_ij with the
% presynaptic afferent x_Ia,j over the full run, searched on the predictive side
% (afferent leads) up to maxLagSec; the lag of the peak |correlation| is shown.
% Antagonist connections (not tested in Fig. 4) and pruned connections are
% left blank. Results are left in the workspace as lagT / lagR.

if ~exist('S','var') || ~exist('W','var')
  error('pubplots_si_7:noData', ...
    'Load the trial first: [S,W,P] = ds_load_trial(root);');
end

fontsize = 12;
labels   = {'SF','SE','EF','EE','BF','BE'};

% dW (ds_load_dw_rule), if in the workspace, gives the rule-computed weight
% change instead of differencing the float32 W
if ~exist('dW','var'), dW = []; end
[lagT, lagR] = gc_lag_estimation(S, W, dWrule=dW);
close(gcf);                                   % its own two-panel figure

L = lagR.ia_peaklag;                          % s; NaN = pruned
C = lagR.ia_peakcor;
[I, J] = ndgrid(1:size(L,1), 1:size(L,2));
L(mod(I + J, 2) == 1) = NaN;                  % antagonists: not tested (Fig. 4)
C(mod(I + J, 2) == 1) = NaN;

lagMax = ceil(max(L(:), [], 'omitnan')*20)/20;   % round up to 0.05 s
cmap   = linspace(0.92, 0.05, 256)' * [1 1 1];   % light grey -> near black

figure('Position', [100 100 430 380], 'Color', 'w');
ax = axes;
him = imagesc(ax, L);
set(him, 'AlphaData', ~isnan(L));             % blank (white) untested cells
ax.Color = 'w';
axis(ax, 'image');
colormap(ax, cmap);  clim(ax, [0 lagMax]);
cb = colorbar(ax);  cb.Label.String = 'Peak lag (s)';
xticks(ax, 1:6);  yticks(ax, 1:6);
xticklabels(ax, labels);  yticklabels(ax, labels);
xlabel(ax, 'Ia afferent');  ylabel(ax, 'Motoneuron');
ax.FontSize = fontsize;  ax.TickLength = [0 0];

% lag value in each tested cell, in a contrasting shade
for k = find(~isnan(L))'
  v = L(k);
  tc = 'k';  if v > 0.55*lagMax, tc = 'w'; end
  text(ax, J(k), I(k), sprintf('%.2f', v), 'HorizontalAlignment', 'center', ...
       'Color', tc, 'FontSize', fontsize-2);
end

fprintf('Ia -> dw peak lag (s), tested connections: min %.3f, max %.3f, mean %.3f\n', ...
        min(L(:),[],'omitnan'), max(L(:),[],'omitnan'), mean(L(:),'omitnan'));
fprintf('peak |corr|: min %.3f, max %.3f\n', ...
        min(abs(C(:)),[],'omitnan'), max(abs(C(:)),[],'omitnan'));
disp(L);
