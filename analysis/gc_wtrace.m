function wij = gc_wtrace(W, i, j)
%GC_WTRACE  Weight trace for connection (i,j), for cell or flat storage.
%
%   wij = GC_WTRACE(W, i, j) returns the time series of W(i,j) as a 1 x N
%   double row vector, accepting either weight representation:
%
%     cell : W{t} is an m x m matrix          (legacy trial_data.mat)
%     flat : W is (m*m) x N, row (i-1)*m + j  (DS_LOAD_TRIAL)
%
%   The flat form is a row slice; the cell form dereferences N cells and is
%   ~60x slower per connection. Returns double so that downstream sums of
%   squares are not accumulated in single.
%
%   See also GC_WSIZE, DS_LOAD_TRIAL.

  if iscell(W)
    wij = arrayfun(@(t) W{t}(i,j), 1:numel(W));
  else
    m = gc_wsize(W);
    wij = double(W((i-1)*m + j, :));
  end
end
