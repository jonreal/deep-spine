function run = ds_find_run(cond_dir, opt)
%DS_FIND_RUN  Run directory in a sweep condition folder, found by parameters.
%   run = DS_FIND_RUN(cond_dir, gain=5, strength=0.2, babble=0.0046) reads
%   each run's custom.txt (DS_READ_PARAMS) under COND_DIR and returns the one
%   run whose Hebbian gain (kappa), max muscle strength (lambda_max) and
%   babble probability (p_b) match. Omitted parameters are not checked.
%   Errors unless exactly one run matches, so a rerun sweep (new run IDs) or a
%   duplicated run can never be picked up silently.
%
%   See also DS_READ_PARAMS, DS_PROCESS_BABBLE.
  arguments
    cond_dir (1,:) char
    opt.gain = []
    opt.strength = []
    opt.babble = []
  end

  want = {'heb_kwa_gai',         opt.gain; ...
          'opt_max_mus_str',     opt.strength; ...
          'sup_kwa_bab_max_pro', opt.babble};
  want = want(~cellfun(@isempty, want(:,2)), :);

  d = dir(cond_dir);
  d = d([d.isdir] & ~startsWith({d.name}, '.'));
  hits = {};
  for k = 1:numel(d)
    path = fullfile(cond_dir, d(k).name);
    if ~isfile(fullfile(path, 'custom.txt')), continue; end
    [keys, vals] = ds_read_params(path);
    match = true;
    for q = 1:size(want,1)
      v = str2double(vals(strcmp(keys, want{q,1})));
      if isempty(v) || abs(v - want{q,2}) > 1e-9
        match = false;
        break
      end
    end
    if match, hits{end+1} = path; end %#ok<AGROW>
  end

  desc = strjoin(cellfun(@(k,v) sprintf('%s=%g', k, v), want(:,1), want(:,2), ...
                         'UniformOutput', false), ', ');
  if isempty(hits)
    error('ds_find_run:none', 'No run in %s with %s.', cond_dir, desc);
  elseif numel(hits) > 1
    error('ds_find_run:many', '%d runs in %s with %s:\n  %s', numel(hits), ...
          cond_dir, desc, strjoin(hits, '\n  '));
  end
  run = hits{1};
end
