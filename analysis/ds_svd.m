function synergy = ds_svd(W)

  [U, S, V] = cellfun(@(Wq) svd(Wq), W, 'UniformOutput', false);
  singular_vals = cellfun(@(s) diag(s), S, 'UniformOutput', false);
  synergy.singular_vals = [singular_vals{:}]';
  synergy.explained_var =  ...
    [synergy.singular_vals'.^2 ./ sum(synergy.singular_vals'.^2)]';
  synergy.sum_of_var = cumsum(synergy.explained_var, 2);

  % exponential of Shannon entropy
  p = synergy.explained_var;
  synergy.eff_rank = exp(-sum(p .* log(p + 1e-12),2));

  %synergy.rank1_mats = cellfun( ...
  %  @(U, S, V) arrayfun( ...
  %    @(i) S(i,i) * U(:,i) * V(:,i)',1:size(S,1), ...
  %    'UniformOutput', false), ...
  %  U, S, V, 'UniformOutput', false);

end
