function trial = load_canonical(data_root, opt)
%LOAD_CANONICAL  Load the canonical run once for Figs 3, 4, S3 and S6.
%   trial = LOAD_CANONICAL(data_root) returns a struct with S, W, P (from
%   DS_LOAD_TRIAL) and dW (from DS_LOAD_DW_RULE, the rule-computed per-step
%   update) for <data_root>/hebbian-canonical/1790298957. Pass it to each figure
%   function as its third argument to avoid re-reading ~26 GB per figure.
%
%   Canonical run: SR0-MR0, kappa 5, lambda_max 0.2, p_b 0.0046, seed 1.
  arguments
    data_root (1,:) char = ''
    opt.dW (1,1) logical = true      % dw_rule.bin is only needed by Figs 4, S6
  end
  data_root = pubfigs_setup(data_root, '');
  root = fullfile(data_root, 'hebbian-canonical', '1790298957');
  [trial.S, trial.W, trial.P] = ds_load_trial(root);
  trial.dW = [];
  if opt.dW, trial.dW = ds_load_dw_rule(root); end
end
