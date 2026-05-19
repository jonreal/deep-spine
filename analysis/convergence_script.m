trial = '../../experiments/hebbian/logs/default/1771903476';

%% Load data
[episode, time, muscle_strength, ~, We] = ...
    extract_data(fullfile(trial, 'data.txt'));


rmean = cell2mat( ...
    arrayfun(@(i) mean(mean(We{i}.^2)),1:numel(We),'UniformOutput',0)');

wc = 0.0001;
[b,a] = butter(1,wc);
r_est = filter(b,a,rmean);


dr_est = 1/wc * (r_est - filter(b,a,r_est));

figure;
  plot(rmean,'k'); hold all;
  plot(r_est,'r')

figure;
  plot(dr_est,'k');
