function [data_root, out_dir] = pubfigs_setup(data_root, out_dir)
%PUBFIGS_SETUP  Shared setup for the paper figure scripts.
%   Adds <repo>/analysis to the path and fills in the default folders:
%     data_root  <repo>/logs                   (hebbian-sweep/, hebbian-canonical/)
%     out_dir    <repo>/analysis/pubfigs/out   (created if missing)
  here = fileparts(mfilename('fullpath'));
  repo = fileparts(fileparts(here));
  addpath(fullfile(repo, 'analysis'));
  if isempty(data_root), data_root = fullfile(repo, 'logs'); end
  if isempty(out_dir),   out_dir   = fullfile(here, 'out');  end
  if ~isfolder(out_dir), mkdir(out_dir); end
end
