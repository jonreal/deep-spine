%% Lag robustness sweep
% Tests incremental kinematic dependence given Ia:
%   restricted:   dW ~ Ia history [+ dW history]
%   unrestricted: dW ~ Ia history + kinematic history [+ dW history]
%
% partialR2_Ia = incremental kinematic variance explained given Ia
% dBIC_Ia      = BIC_unrestricted - BIC_restricted
%                negative values favor adding kinematics

clearvars -except S W

fs = 200;

% Use integer sample counts so every model is distinct.
lag_samples = 2:10:100;                 % 10 ms to 500 ms
lag_range   = lag_samples / fs;
nLag        = numel(lag_range);

method = 'full';

mask = make_mask();

%% Run all models first

PR2 = nan(6,6,nLag);
DBIC = nan(6,6,nLag);

for k = 1:nLag
    fprintf('Lag %3d/%3d: %3d samples = %.3f s\n', ...
        k, nLag, lag_samples(k), lag_range(k));

    R = gc_batch( ...
        S, W, ...
        lag     = lag_range(k), ...
        feature = method, ...
        plot    = false);

    assert(isequal(size(R.partialR2_Ia), [6 6]), ...
        'Unexpected size for R.partialR2_Ia.');
    assert(isequal(size(R.dBIC_Ia), [6 6]), ...
        'Unexpected size for R.dBIC_Ia.');

    PR2(:,:,k)  = R.partialR2_Ia;
    DBIC(:,:,k) = R.dBIC_Ia;
end

%% Hold group membership fixed across the sweep

% A connection is retained only if both statistics are finite at every lag.
common_valid = all(isfinite(PR2),3) & all(isfinite(DBIC),3);

syn_mask = mask.synergy & common_valid;
ref_mask = mask.reflex  & common_valid;

nSyn = nnz(syn_mask);
nRef = nnz(ref_mask);

fprintf('\nConnections retained over the full lag sweep:\n');
fprintf('  synergy: %d / %d\n', nSyn, nnz(mask.synergy));
fprintf('  reflex:  %d / %d\n', nRef, nnz(mask.reflex));

if nSyn == 0 || nRef == 0
    error(['No valid connections remain in one group. Inspect which lag ' ...
           'values produce NaNs or restrict the sweep range.']);
end

% Convert 6x6xN arrays to connection-by-lag matrices.
PR2_flat  = reshape(PR2,  [], nLag);
DBIC_flat = reshape(DBIC, [], nLag);

PR2_syn  = PR2_flat(find(syn_mask(:)), :);
PR2_ref  = PR2_flat(find(ref_mask(:)), :);
DBIC_syn = DBIC_flat(find(syn_mask(:)), :);
DBIC_ref = DBIC_flat(find(ref_mask(:)), :);

%% Summary statistics across connections

mean_pr2_syn = mean(PR2_syn,1);
mean_pr2_ref = mean(PR2_ref,1);

med_pr2_syn = median(PR2_syn,1);
med_pr2_ref = median(PR2_ref,1);

mean_bic_syn = mean(DBIC_syn,1);
mean_bic_ref = mean(DBIC_ref,1);

med_bic_syn = median(DBIC_syn,1);
med_bic_ref = median(DBIC_ref,1);

% Fraction of connections where unrestricted model is BIC-preferred.
frac_bic_syn = mean(DBIC_syn < 0,1);
frac_bic_ref = mean(DBIC_ref < 0,1);

% Direct mechanistic contrasts.
delta_pr2 = mean_pr2_syn - mean_pr2_ref;
delta_bic = mean_bic_syn - mean_bic_ref;

% Positive delta_pr2 means stronger kinematic dependence in synergies.
% Negative delta_bic means BIC favors kinematics more strongly in synergies.

%% Save numerical summary

lag_summary = table( ...
    lag_samples(:), ...
    lag_range(:), ...
    mean_pr2_syn(:), ...
    mean_pr2_ref(:), ...
    med_pr2_syn(:), ...
    med_pr2_ref(:), ...
    mean_bic_syn(:), ...
    mean_bic_ref(:), ...
    med_bic_syn(:), ...
    med_bic_ref(:), ...
    frac_bic_syn(:), ...
    frac_bic_ref(:), ...
    delta_pr2(:), ...
    delta_bic(:), ...
    'VariableNames', { ...
        'lag_samples', 'lag_seconds', ...
        'mean_pr2_synergy', 'mean_pr2_reflex', ...
        'median_pr2_synergy', 'median_pr2_reflex', ...
        'mean_dBIC_synergy', 'mean_dBIC_reflex', ...
        'median_dBIC_synergy', 'median_dBIC_reflex', ...
        'fraction_dBIC_negative_synergy', ...
        'fraction_dBIC_negative_reflex', ...
        'delta_pr2_synergy_minus_reflex', ...
        'delta_dBIC_synergy_minus_reflex'});

%% Figure 1: group summaries

figure('Name','Lag sweep: group summaries');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(lag_range, mean_pr2_syn, 'LineWidth',2);
hold on;
plot(lag_range, mean_pr2_ref, 'LineWidth',2);
plot(lag_range, med_pr2_syn, '--', 'LineWidth',1);
plot(lag_range, med_pr2_ref, '--', 'LineWidth',1);
xlabel('History window (s)');
ylabel('Partial R^2_{kin|Ia}');
title('Incremental kinematic prediction');
legend( ...
    'Synergy mean','Reflex mean', ...
    'Synergy median','Reflex median', ...
    'Location','best');
grid on;

nexttile;
plot(lag_range, mean_bic_syn, 'LineWidth',2);
hold on;
plot(lag_range, mean_bic_ref, 'LineWidth',2);
plot(lag_range, med_bic_syn, '--', 'LineWidth',1);
plot(lag_range, med_bic_ref, '--', 'LineWidth',1);
yline(0,':');
xlabel('History window (s)');
ylabel('\DeltaBIC_{kin|Ia}');
title('Penalized evidence for kinematics');
legend( ...
    'Synergy mean','Reflex mean', ...
    'Synergy median','Reflex median', ...
    'Location','best');
grid on;

nexttile;
plot(lag_range, frac_bic_syn, 'LineWidth',2);
hold on;
plot(lag_range, frac_bic_ref, 'LineWidth',2);
yline(0.5,':');
ylim([0 1]);
xlabel('History window (s)');
ylabel('Fraction with \DeltaBIC < 0');
title('Connections favoring kinematic model');
legend('Synergy','Reflex','Location','best');
grid on;

nexttile;
plot(lag_range, delta_pr2, 'LineWidth',2);
hold on;
yline(0,':');
xlabel('History window (s)');
ylabel('\Delta partial R^2');
title('Synergy minus reflex');
grid on;

sgtitle('Kinematic dependence conditional on Ia');

%% Figure 2: direct group contrasts

figure('Name','Lag sweep: synergy-reflex contrasts');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(lag_range, delta_pr2, 'LineWidth',2);
hold on;
yline(0,':');
xlabel('History window (s)');
ylabel('\Delta partial R^2_{kin|Ia}');
title('Synergy minus reflex: positive supports greater kinematic dependence');
grid on;

nexttile;
plot(lag_range, delta_bic, 'LineWidth',2);
hold on;
yline(0,':');
xlabel('History window (s)');
ylabel('\Delta\DeltaBIC_{kin|Ia}');
title('Synergy minus reflex: negative favors kinematics more in synergies');
grid on;

%% Figure 3: individual connection traces

figure('Name','Lag sweep: individual connections');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(lag_range, PR2_syn', 'LineWidth',0.5);
hold on;
plot(lag_range, mean_pr2_syn, 'k', 'LineWidth',2.5);
xlabel('History window (s)');
ylabel('Partial R^2_{kin|Ia}');
title(sprintf('Synergy connections, n=%d',nSyn));
grid on;

nexttile;
plot(lag_range, PR2_ref', 'LineWidth',0.5);
hold on;
plot(lag_range, mean_pr2_ref, 'k', 'LineWidth',2.5);
xlabel('History window (s)');
ylabel('Partial R^2_{kin|Ia}');
title(sprintf('Reflex connections, n=%d',nRef));
grid on;

nexttile;
plot(lag_range, DBIC_syn', 'LineWidth',0.5);
hold on;
plot(lag_range, mean_bic_syn, 'k', 'LineWidth',2.5);
yline(0,':');
xlabel('History window (s)');
ylabel('\DeltaBIC_{kin|Ia}');
title('Synergy connections');
grid on;

nexttile;
plot(lag_range, DBIC_ref', 'LineWidth',0.5);
hold on;
plot(lag_range, mean_bic_ref, 'k', 'LineWidth',2.5);
yline(0,':');
xlabel('History window (s)');
ylabel('\DeltaBIC_{kin|Ia}');
title('Reflex connections');
grid on;

%% Optional: identify broad candidate lag region

% Example descriptive criterion:
%   synergy partial R2 exceeds reflex
%   synergy mean dBIC is negative
%   at least half of synergy connections favor adding kinematics
candidate = ...
    delta_pr2 > 0 & ...
    mean_bic_syn < 0 & ...
    frac_bic_syn >= 0.5;

if any(candidate)
    fprintf('\nCandidate lag range under descriptive criteria:\n');
    fprintf('  %.3f to %.3f s\n', ...
        min(lag_range(candidate)), max(lag_range(candidate)));
else
    fprintf('\nNo lag satisfies all descriptive criteria.\n');
end


function mask = make_mask()
% Motif group masks built from the ds_motif basis.
%
% mask.reflex:
%   block-diagonal homonymous/reflex connections
%
% mask.synergy:
%   off-diagonal symmetric and antisymmetric synergy connections

    m = 6;

    S_template = [ ...
        1 0 zeros(1,m-2); ...
        0 1 zeros(1,m-2); ...
        zeros(m-2,m)];

    A_template = [ ...
        0 1 zeros(1,m-2); ...
        1 0 zeros(1,m-2); ...
        zeros(m-2,m)];

    bS = arrayfun(@(i) arrayfun(@(j) ...
        reshape(circshift(S_template,[i j]),[],1), ...
        0:2:(m-2), ...
        'UniformOutput',false), ...
        0:2:(m-2), ...
        'UniformOutput',false);

    bA = arrayfun(@(i) arrayfun(@(j) ...
        reshape(circshift(A_template,[i j]),[],1), ...
        0:2:(m-2), ...
        'UniformOutput',false), ...
        0:2:(m-2), ...
        'UniformOutput',false);

    bR = cell2mat(arrayfun(@(i) ...
        bS{i}{i}, ...
        1:(m/2), ...
        'UniformOutput',false));

    % Remove block-diagonal symmetric motifs from synergy set.
    for i = 1:(m/2)
        bS{i}(i) = [];
    end

    bS_mat = cell2mat(cellfun(@(x) ...
        [x{:}], bS, 'UniformOutput',false));

    bA_mat = cell2mat(cellfun(@(x) ...
        [x{:}], bA, 'UniformOutput',false));

    mask.reflex = reshape( ...
        any(bR > 0,2), ...
        m,m);

    mask.synergy = reshape( ...
        any([bS_mat,bA_mat] > 0,2), ...
        m,m);
end
