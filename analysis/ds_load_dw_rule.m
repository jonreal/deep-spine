function dW = ds_load_dw_rule(root, opt)
%DS_LOAD_DW_RULE  Load the rule-computed per-step weight updates.
%   dW = DS_LOAD_DW_RULE(root) reads <root>/dw_rule.bin, written by
%   export_dw_rule.py, as a flat (m*m) x N single matrix in the same layout
%   as W from DS_LOAD_TRIAL (row (i-1)*m + j). Column t is the update applied
%   at step t before float32 rounding, so W(:,t) = single(W(:,t-1) + dW(:,t)).
%   Pass it to GC_BATCH / GC_RUN / GC_WEIGHT_EVENTS as dWrule=dW.
%
%   See also DS_LOAD_TRIAL, GC_WEIGHT_EVENTS.
  arguments
    root (1,:) char
    opt.nMuscle (1,1) double = 6
  end
  nW  = opt.nMuscle^2;
  fid = fopen(fullfile(root, 'dw_rule.bin'), 'r');
  if fid < 0
    error('ds_load_dw_rule:missing', ...
      'No dw_rule.bin in %s; run analysis/export_dw_rule.py first.', root);
  end
  c  = onCleanup(@() fclose(fid));
  dW = fread(fid, [nW, Inf], '*single');
end
