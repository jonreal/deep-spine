function make_all(data_root, out_dir)
%MAKE_ALL  Regenerate every paper figure and Table S1.
%   MAKE_ALL() reads <repo>/logs and writes to <repo>/analysis/pubfigs/out.
%   The canonical run is loaded once and shared by Figs 3, 4, S3 and S6.
%   Fig S5 needs out/si_plant.mat first: python analysis/pubfigs/export_plant.py
  arguments
    data_root (1,:) char = ''
    out_dir (1,:) char = ''
  end
  [data_root, out_dir] = pubfigs_setup(data_root, out_dir);

  % sweep summary tables
  fig2_motif(data_root, out_dir);
  fig5_sweep(data_root, out_dir);
  figS1S2_kappa(data_root, out_dir);
  figS4_sweep_mr1(data_root, out_dir);
  tableS1_motif(data_root, out_dir);
  figS5_plant(data_root, out_dir);

  % canonical run
  trial = load_canonical(data_root);
  fig3_grid(data_root, out_dir, trial);
  fig4_granger(data_root, out_dir, trial);
  figS3_timehistory(data_root, out_dir, trial);
  figS6_ia_lag(data_root, out_dir, trial);
end
