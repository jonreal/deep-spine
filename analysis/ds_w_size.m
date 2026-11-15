function [m, n, N] = ds_w_size(W)
%DS_W_SIZE  Connectivity dimensions and timestep count, for cell or flat W.
%
%   [m, n, N] = DS_W_SIZE(W) returns the m x n connectivity shape and the
%   number of timesteps N, accepting either representation:
%
%     cell : W{t} is m x n, N = numel(W)
%     flat : W is (m*n) x N with m == n (square connectivity assumed)
%
%   See also DS_W_TRACE, DS_LOAD_TRIAL.

  if iscell(W)
    [m, n] = size(W{1});
    N = numel(W);
  else
    N = size(W, 2);
    m = sqrt(size(W, 1));
    if m ~= floor(m)
      error('ds_w_size:notSquare', ...
        ['Flat W has %d rows, which is not a square number. Flat storage ' ...
         'assumes square connectivity; pass a cell array otherwise.'], ...
        size(W,1));
    end
    n = m;
  end
end
