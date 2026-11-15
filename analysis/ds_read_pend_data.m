function S = ds_read_pend_data(root)
% DS_READ_PEND_DATA  Read binary env.txt from arm26_pendulum simulation.
%
%   S = ds_read_pend_data(ROOT) reads ROOT/env.txt and returns a struct
%   with fields matching the arm26_pendulum _write_data_to_logfile layout.
%
%   Each field is a cell array indexed by episode. Access:
%     S.qpos{k}       — [T_k x 3] matrix for episode k
%     S.act_Ia{5}     — Ia activations for episode 5
%     S.num_eps        — total number of episodes
%
%   Column layout (all float64):
%     eps(1) time(1) qpos(3) qvel(3) length(6) velocity(6) force(6)
%     act_II(6) act_Ia(6) act_MN(6) drive(6) act_muscle(6) act(6)
%     visual(2) action(6)

  colsizes = [1, 1, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 6];
  ncols = sum(colsizes);

  fid = fopen(fullfile(root, 'env.txt'), 'rb');
  assert(fid ~= -1, 'Could not open %s/env.txt', root);
  A = fread(fid, [ncols, Inf], 'double')';
  fclose(fid);

  % Unpack columns — offsets computed from cumulative colsizes
  c = cumsum([0, colsizes]);
  idx = @(k) (c(k)+1):c(k+1);

  fields = {'time','qpos','qvel','length','velocity','force', ...
            'act_II','act_Ia','act_MN','drive','act_muscle','act', ...
            'visual','action'};

  % Split by episode
  eps_col = A(:, idx(1));
  ep_ids = unique(eps_col);
  num_eps = numel(ep_ids);

  % Preallocate cell arrays
  for f = 1:numel(fields)
    S.(fields{f}) = cell(num_eps, 1);
  end

  for k = 1:num_eps
    mask = eps_col == ep_ids(k);
    for f = 1:numel(fields)
      S.(fields{f}){k} = A(mask, idx(f + 1));
    end
  end

  S.num_eps = num_eps;

end
