%% Convergence Lab
% Prototype convergence detection on a single trial
clear; clc; close all;

trial = '../../experiments/hebbian/logs/default/1771903476';

%% Load data
[episode, time, muscle_strength, ~, We] = ...
    extract_data(fullfile(trial, 'data.txt'));

%% Build synergy masks
n = 6;
mR = diag(repmat([1,0],1,n/2)) + diag(repmat([0,1],1,n/2));
mSf = repmat([1,0; 0,0], n/2, n/2) - diag(repmat([1,0],1,n/2));
mSe = repmat([0,0; 0,1], n/2, n/2) - diag(repmat([0,1],1,n/2));
mS = mSf + mSe;
mAf = repmat([0,0; 1,0], n/2, n/2);
mAe = repmat([0,1; 0,0], n/2, n/2);
mA = mAf + mAe;

%% Compute signals per sample
N = numel(We);
wp    = zeros(N,1);  % RMS weight (current convergence signal)
R_a   = zeros(N,1);  % Reflex motif
S_a   = zeros(N,1);  % Synergy motif
A_a   = zeros(N,1);  % Anti-synergy motif

for k = 1:N
    W = We{k};
    wp(k)  = sqrt(mean(W(:).^2));
    R_a(k) = sum(W .* mR, 'all') / sum(mR, 'all');
    S_a(k) = sum(W .* mS, 'all') / sum(mS, 'all');
    A_a(k) = sum(W .* mA, 'all') / sum(mA, 'all');
end

eps_ids = unique(episode);
n_eps = numel(eps_ids);

%% Compute per-episode window stats for each signal
signals = {wp, R_a, S_a, A_a};
signal_names = {'RMS(W)', 'Reflex', 'Synergy', 'Anti'};
n_sig = numel(signals);

% Per-episode: mean and std of each signal
ep_mu  = zeros(n_eps, n_sig);
ep_sig = zeros(n_eps, n_sig);
ep_strength = zeros(n_eps, 1);

for i = 1:n_eps
    idx = episode == eps_ids(i);
    ep_strength(i) = muscle_strength(find(idx, 1, 'last'));
    for s = 1:n_sig
        x = signals{s}(idx);
        ep_mu(i,s) = mean(x);
        ep_sig(i,s) = std(x);
    end
end

%% Run WindowStationarity on each signal
M = 3;         % number of windows to compare
epsilon = 1e-2;  % rtol_mu
atol_sig = 1e-3; % atol_sig

% Storage
mu_span  = nan(n_eps, n_sig);
sig_max  = nan(n_eps, n_sig);

for s = 1:n_sig
    mus  = zeros(M, 1); % circular buffer
    sigs = zeros(M, 1);
    ptr = 0;
    count = 0;
    for i = 1:n_eps
        ptr = mod(ptr, M) + 1;
        mus(ptr) = ep_mu(i, s);
        sigs(ptr) = ep_sig(i, s);
        count = count + 1;
        if count >= M
            mu_ref = max(1e-12, mean(abs(mus)));
            mu_span(i, s) = (max(mus) - min(mus)) / mu_ref;
            sig_max(i, s) = max(sigs);
        end
    end
end

converged_mu  = mu_span < epsilon;
converged_sig = sig_max < atol_sig;
converged     = converged_mu & converged_sig;

%% Figure 1: Raw signals over time
figure('Position', [50 50 1400 800]);

for s = 1:n_sig
    subplot(n_sig, 1, s);
    plot(signals{s}, '.', 'MarkerSize', 1);
    ylabel(signal_names{s});
    axis tight;
    if s == 1, title('Raw signals (per sample)'); end
    if s < n_sig, xticklabels({}); end
end
xlabel('Sample index');

%% Figure 2: Per-episode stats + convergence criteria
figure('Position', [50 50 1400 900]);
colors = lines(n_sig);

% Panel 1: Episode means
subplot(4,1,1);
for s = 1:n_sig
    plot(eps_ids, ep_mu(:,s), '-', 'Color', colors(s,:), 'LineWidth', 1); hold on;
end
legend(signal_names, 'Location', 'best');
ylabel('Window mean (\mu)');
title('Per-episode convergence diagnostics');
xticklabels({});
axis tight;

% Panel 2: Episode std
subplot(4,1,2);
for s = 1:n_sig
    plot(eps_ids, ep_sig(:,s), '-', 'Color', colors(s,:), 'LineWidth', 1); hold on;
end
yline(atol_sig, 'k--', 'atol\_sig', 'LineWidth', 1.5);
ylabel('Window std (\sigma)');
xticklabels({});
axis tight;

% Panel 3: mu_span (relative drift)
subplot(4,1,3);
for s = 1:n_sig
    plot(eps_ids, mu_span(:,s), '-', 'Color', colors(s,:), 'LineWidth', 1); hold on;
end
yline(epsilon, 'k--', '\epsilon', 'LineWidth', 1.5);
ylabel('mu\_span (rel)');
set(gca, 'YScale', 'log');
xticklabels({});
axis tight;

% Panel 4: sig_max (absolute)
subplot(4,1,4);
for s = 1:n_sig
    plot(eps_ids, sig_max(:,s), '-', 'Color', colors(s,:), 'LineWidth', 1); hold on;
end
yline(atol_sig, 'k--', 'atol\_sig', 'LineWidth', 1.5);
ylabel('sig\_max (abs)');
set(gca, 'YScale', 'log');
xlabel('Episode');
axis tight;

%% Figure 3: std(mus)/mean(mus) — proposed replacement for mu_span
M_test = [10, 20, 50];
epsilon_test = 1e-2;

figure('Position', [50 50 1400 900]);

for mi = 1:numel(M_test)
    Mw = M_test(mi);
    cv_mu = nan(n_eps, n_sig);  % coefficient of variation of window means

    for s = 1:n_sig
        for i = Mw:n_eps
            window = ep_mu(i-Mw+1:i, s);
            cv_mu(i, s) = std(window) / max(1e-12, abs(mean(window)));
        end
    end

    subplot(numel(M_test), 1, mi);
    for s = 1:n_sig
        plot(eps_ids, cv_mu(:,s), '-', 'Color', colors(s,:), 'LineWidth', 1);
        hold on;
    end
    yline(epsilon_test, 'k--', '\epsilon', 'LineWidth', 1.5);
    ylabel(sprintf('std/mean (M=%d)', Mw));
    set(gca, 'YScale', 'log');
    axis tight;
    ylim([1e-4 1]);
    if mi == 1
        title('Proposed: std(\mu) / mean(|\mu|) over sliding window');
        legend(signal_names, 'Location', 'best');
    end
    if mi < numel(M_test), xticklabels({}); end
end
xlabel('Episode');

%% Figure 4: First convergence episode — old vs proposed
figure('Position', [50 50 1000 500]);

% Old method (range, M=3)
subplot(1,2,1);
for s = 1:n_sig
    first_conv = find(converged(:,s), 1, 'first');
    if isempty(first_conv)
        bar(s, n_eps, 'FaceColor', [0.8 0.2 0.2]); hold on;
        text(s, n_eps + 50, 'never', 'HorizontalAlignment', 'center');
    else
        bar(s, eps_ids(first_conv), 'FaceColor', colors(s,:)); hold on;
        text(s, eps_ids(first_conv) + 50, num2str(eps_ids(first_conv)), ...
             'HorizontalAlignment', 'center');
    end
end
set(gca, 'XTick', 1:n_sig, 'XTickLabel', signal_names);
ylabel('First convergence episode');
title(sprintf('Old: range, M=%d, \\epsilon=%.0e, atol=%.0e', M, epsilon, atol_sig));

% Proposed method (std, M=20, mu-only)
Mw_proposed = 20;
subplot(1,2,2);
for s = 1:n_sig
    cv = nan(n_eps, 1);
    for i = Mw_proposed:n_eps
        window = ep_mu(i-Mw_proposed+1:i, s);
        cv(i) = std(window) / max(1e-12, abs(mean(window)));
    end
    first_conv = find(cv < epsilon_test, 1, 'first');
    if isempty(first_conv)
        bar(s, n_eps, 'FaceColor', [0.8 0.2 0.2]); hold on;
        text(s, n_eps + 50, 'never', 'HorizontalAlignment', 'center');
    else
        bar(s, eps_ids(first_conv), 'FaceColor', colors(s,:)); hold on;
        text(s, eps_ids(first_conv) + 50, num2str(eps_ids(first_conv)), ...
             'HorizontalAlignment', 'center');
    end
end
set(gca, 'XTick', 1:n_sig, 'XTickLabel', signal_names);
ylabel('First convergence episode');
title(sprintf('Proposed: std/mean, M=%d, \\epsilon=%.0e (\\mu only)', ...
    Mw_proposed, epsilon_test));
