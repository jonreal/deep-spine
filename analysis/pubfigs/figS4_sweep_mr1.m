function figS4_sweep_mr1(data_root, out_dir)
%FIGS4_SWEEP_MR1  SI figure: Fig 5 (embodiment sweep) for the MR1 bodies.
% Identical layout, operating points and styling to FIG5_SWEEP; only the rows
% differ (uniform mass 1.5:1.5 instead of MR0). Writes <out_dir>/si-sweep-mr1.pdf.
arguments
  data_root (1,:) char = ''
  out_dir (1,:) char = ''
end
fig5_sweep(data_root, out_dir, tags={'SR0-MR1','SR1-MR1','SR2-MR1'}, ...
           outFile='si-sweep-mr1.pdf');
end
