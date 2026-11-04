function [S, W, P] = ds_load_trial(root, opt)
%DS_LOAD_TRIAL  Load a babble/Hebbian trial from its raw binary logs.
%
%   [S, W, P] = DS_LOAD_TRIAL(root) reads env.txt, hebbian.txt and data.txt
%   from the trial directory ROOT and returns
%
%     S - struct of per-step signals, time-major (N x width), see LAYOUT below
%     W - Ia weights, FLAT storage: (m*m) x N, row (i-1)*m + j is W(i,j).
%         Use GC_WTRACE(W,i,j) rather than indexing directly.
%     P - metadata: sampling rate, episode boundaries, the 1 Hz muscle-strength
%         schedule, and the run's config/custom JSON.
%
%   W is deliberately NOT a cell array. A cell of N 6x6 matrices costs ~400 B
%   per timestep and makes the per-connection extraction O(N) through a cell
%   dereference; at N = 36e6 that is ~14 GB and ~30 s per connection. The flat
%   form is 5.2 GB and extraction is a row slice.
%
%   Name-value options:
%     signals    - cellstr subset of S fields to load ({} = all). 'time' is
%                  always loaded; it carries the episode boundaries.
%     precision  - 'single' (default) or 'double' for S. env.txt is written as
%                  float64, but single halves the footprint and fitlm accepts
%                  it; GC_RUN casts the regression features back to double.
%     fs         - sampling rate in Hz (default 200), recorded in P.
%     chunk      - rows per read (default 2e6); peak memory is the output plus
%                  one chunk rather than two full copies.
%     maxRows    - read only the first maxRows samples (default Inf). Use a
%                  small value to smoke-test against a big trial.
%     nMuscle    - muscle count (default 6); W is nMuscle^2 rows.
%     save       - if true, save S/W/P to fullfile(root, savefile).
%     savefile   - default 'trial_data.mat'.
%     verbose    - print progress (default true).
%
%   Example:
%     root = '../experiments/hebbian/logs/example-1/1787603818';
%     [S,W,P] = ds_load_trial(root, save=true);
%
%   See also GC_WTRACE, GC_WSIZE, EXTRACT_DATA.

  arguments
    root (1,:) char
    opt.signals (1,:) cell = {}
    opt.precision (1,:) char {mustBeMember(opt.precision,{'single','double'})} = 'single'
    opt.fs (1,1) double = 200
    opt.chunk (1,1) double = 2e6
    opt.maxRows (1,1) double = Inf
    opt.nMuscle (1,1) double = 6
    opt.save (1,1) logical = false
    opt.savefile (1,:) char = 'trial_data.mat'
    opt.verbose (1,1) logical = true
  end

  % Column layout of env.txt, matching arm26_babble._write_data_to_logfile.
  layout = { ...
      'time',        1; ...
      'qpos',        2; ...
      'qvel',        2; ...
      'len',         6; ...
      'vel',         6; ...
      'force',       6; ...
      'act_IIa',     6; ...
      'act_Ia',      6; ...
      'drive',       6; ...
      'act_MN',      6; ...
      'act_muscle',  6; ...
      'act',         6};

  names   = layout(:,1);
  widths  = cell2mat(layout(:,2));
  offs    = cumsum([1; widths(1:end-1)]);
  ncols   = sum(widths);

  if isempty(opt.signals)
    keep = true(numel(names),1);
  else
    keep = ismember(names, opt.signals);
    unknown = setdiff(opt.signals, names);
    if ~isempty(unknown)
      error('ds_load_trial:unknownSignal', ...
            'Unknown signal(s): %s', strjoin(unknown, ', '));
    end
  end
  keep(strcmp(names,'time')) = true;   % always needed for episode boundaries

  nW      = opt.nMuscle^2;
  prec    = opt.precision;
  vb      = opt.verbose;

  env_path = fullfile(root, 'env.txt');
  heb_path = fullfile(root, 'hebbian.txt');

  % ---- sample counts, from file size ------------------------------------
  N_env = count_rows(env_path, ncols, 8, 'env.txt');
  N_heb = count_rows(heb_path, nW,    4, 'hebbian.txt');

  N = min([N_env, N_heb, opt.maxRows]);

  if N_env ~= N_heb && vb
    fprintf(['ds_load_trial: env.txt has %d rows, hebbian.txt has %d ' ...
             '(differ by %d); truncating both to %d.\n'], ...
             N_env, N_heb, abs(N_env-N_heb), N);
  end
  if vb
    fprintf('ds_load_trial: %s\n', root);
    fprintf('  N = %d samples (%.1f s at %g Hz)\n', N, N/opt.fs, opt.fs);
  end

  % ---- env.txt, chunked --------------------------------------------------
  S = struct();
  for k = 1:numel(names)
    if keep(k)
      S.(names{k}) = zeros(N, widths(k), prec);
    end
  end

  t0  = tic;
  fid = fopen(env_path, 'rb');
  if fid < 0, error('ds_load_trial:open', 'Cannot open %s', env_path); end
  cleaner = onCleanup(@() fclose(fid));

  ptr = 1;
  while ptr <= N
    m = min(opt.chunk, N - ptr + 1);
    A = fread(fid, [ncols, m], '*double');
    got = size(A,2);
    if got == 0, break; end
    idx = ptr:(ptr+got-1);
    for k = 1:numel(names)
      if ~keep(k), continue; end
      cols = offs(k):(offs(k)+widths(k)-1);
      S.(names{k})(idx,:) = cast(A(cols,:).', prec);
    end
    ptr = ptr + got;
    if vb
      fprintf('  env.txt %5.1f%%\r', 100*(ptr-1)/N);
    end
  end
  clear cleaner A
  if vb
    fprintf('  env.txt   read in %.1f s (%d of %d cols kept, %s)\n', ...
            toc(t0), sum(widths(keep)), ncols, prec);
  end

  % ---- hebbian.txt -------------------------------------------------------
  t1  = tic;
  fid = fopen(heb_path, 'rb');
  if fid < 0, error('ds_load_trial:open', 'Cannot open %s', heb_path); end
  cleaner = onCleanup(@() fclose(fid));
  W = fread(fid, [nW, N], '*single');   % '*single' keeps it single-precision
  clear cleaner
  if vb
    fprintf('  hebbian.txt read in %.1f s (%d x %d single, %.2f GB)\n', ...
            toc(t1), size(W,1), size(W,2), numel(W)*4/1e9);
  end

  % ---- metadata ----------------------------------------------------------
  P = struct();
  P.root      = root;
  P.fs        = opt.fs;
  P.N         = N;
  P.nMuscle   = opt.nMuscle;
  P.precision = prec;

  % Episode boundaries: env.txt column 1 is data.time, which resets each episode.
  tt          = double(S.time);
  is_start    = [true; diff(tt) < 0];
  P.ep_start  = find(is_start);
  P.n_episode = numel(P.ep_start);
  S.episode   = cast(cumsum(is_start), pick_int(P.n_episode));

  if vb
    fprintf('  %d episodes detected (median %.1f s)\n', ...
            P.n_episode, median(diff([P.ep_start; N+1]))/opt.fs);
  end

  % 1 Hz summary log: episode, time, muscle_strength, W(36).
  dpath = fullfile(root, 'data.txt');
  if isfile(dpath)
    nc1 = 3 + nW;
    M   = count_rows(dpath, nc1, 8, 'data.txt');
    fid = fopen(dpath, 'rb');
    cleaner = onCleanup(@() fclose(fid));
    A1  = fread(fid, [nc1, M], '*double');
    clear cleaner
    P.episode_1hz        = A1(1,:).';
    P.time_1hz           = A1(2,:).';
    P.muscle_strength    = A1(3,:).';
    P.W_1hz              = single(A1(4:end,:));   % same flat convention as W
    if vb
      fprintf('  data.txt  %d rows, muscle_strength %.4g -> %.4g\n', ...
              M, P.muscle_strength(1), P.muscle_strength(end));
    end
  end

  P.config = read_json(fullfile(root, 'config.json'));
  P.custom = read_json(fullfile(root, 'custom.txt'));

  % ---- save --------------------------------------------------------------
  if opt.save
    out = fullfile(root, opt.savefile);
    if vb
      bytes = numel(W)*4;
      f = fieldnames(S);
      for k = 1:numel(f), x = S.(f{k}); bytes = bytes + numel(x)*sizeof(x); end
      fprintf('  saving ~%.1f GB to %s ...\n', bytes/1e9, out);
    end
    t2 = tic;
    save(out, 'S', 'W', 'P', '-v7.3', '-nocompression');
    if vb, fprintf('  saved in %.1f s\n', toc(t2)); end
  end
end

% ---------------------------------------------------------------------------
function N = count_rows(path, ncols, bytes_per_elem, label)
  d = dir(path);
  if isempty(d)
    error('ds_load_trial:missing', 'Missing %s at %s', label, path);
  end
  row_bytes = ncols * bytes_per_elem;
  N = d.bytes / row_bytes;
  if mod(d.bytes, row_bytes) ~= 0
    warning('ds_load_trial:partialRow', ...
        ['%s is %d bytes, not a whole multiple of %d (%d cols x %d B). ' ...
         'Truncating to %d complete rows.'], ...
        label, d.bytes, row_bytes, ncols, bytes_per_elem, floor(N));
  end
  N = floor(N);
end

function s = read_json(path)
  s = struct();
  if ~isfile(path), return; end
  txt = fileread(path);
  try
    s = jsondecode(txt);
  catch
    s = txt;   % not JSON (e.g. an older key=value custom.txt); keep it raw
  end
end

function c = pick_int(n)
  if     n <= intmax('uint16'), c = 'uint16';
  elseif n <= intmax('uint32'), c = 'uint32';
  else,                         c = 'double';
  end
end

function n = sizeof(x)
  switch class(x)
    case {'double','uint64','int64'}, n = 8;
    case {'single','uint32','int32'}, n = 4;
    case {'uint16','int16'},          n = 2;
    otherwise,                        n = 1;
  end
end
