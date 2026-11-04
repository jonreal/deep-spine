function [m, n, N] = gc_wsize(W)
%GC_WSIZE  Connectivity dimensions and timestep count, for cell or flat W.
%
%   [m, n, N] = GC_WSIZE(W) returns the m x n connectivity shape and the
%   number of timesteps N, accepting either representation:
%
%     cell : W{t} is m x n, N = numel(W)
%     flat : W is (m*n) x N with m == n (square connectivity assumed)
%
%   See also GC_WTRACE, DS_LOAD_TRIAL.

  if iscell(W)
    [m, n] = size(W{1});
    N = numel(W);
  else
    N = size(W, 2);
    m = sqrt(size(W, 1));
    if m ~= floor(m)
      error('gc_wsize:notSquare', ...
        ['Flat W has %d rows, which is not a square number. Flat storage ' ...
         'assumes square connectivity; pass a cell array otherwise.'], ...
        size(W,1));
    end
    n = m;
  end
end
