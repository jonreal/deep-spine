function tableS1_motif(data_root, out_dir)
%TABLES1_MOTIF  SI table: motif coefficients at the Fig 5 operating points.
%
% For every body (3 SR x 2 MR) and each Fig 5 operating point, prints the nine
% pool coefficients (r, s, a for the S, E, B pools) from the sweep summary
% tables at kappa = 5, and writes the LaTeX table body to
% <out_dir>/si-motif-table.tex.

arguments
  data_root (1,:) char = ''
  out_dir (1,:) char = ''
end
[data_root, out_dir] = pubfigs_setup(data_root, out_dir);

root     = fullfile(data_root, 'hebbian-sweep');
outFile  = fullfile(out_dir, 'si-motif-table.tex');
heb_gain = 5;

tags  = {'SR0-MR0','SR1-MR0','SR2-MR0','SR0-MR1','SR1-MR1','SR2-MR1'};
pts   = struct('sym', {'\ding{73}', '$\triangle$', '$\Diamond$'}, ...
               'pb',  {0.0046, 0.0129, 0.0028}, ...
               'lam', {0.2,    0.5,    0.9});
flds  = {'rS_mean','rE_mean','rB_mean','sS_mean','sE_mean','sB_mean', ...
         'aS_mean','aE_mean','aB_mean'};

lines = {};
for p = 1:numel(pts)
  lines{end+1} = sprintf('\\multicolumn{10}{l}{%s\\ $p_{\\mathrm b}=%g$, $\\lambda_{\\max}=%g$}\\\\', ...
                         pts(p).sym, pts(p).pb, pts(p).lam); %#ok<SAGROW>
  for b = 1:numel(tags)
    L = load(fullfile(root, tags{b}, 'analysis', 'summary', 'summary_table.mat'), 'summary_table');
    T = L.summary_table;
    k = find(abs(T.opt_max_mus_str - pts(p).lam) < 1e-6 & ...
             abs(T.sup_kwa_bab_max_pro - pts(p).pb) < 1e-6 & ...
             T.heb_kwa_gai == heb_gain, 1);
    if isempty(k)
      vals = repmat({'--'}, 1, numel(flds));
    else
      vals = cellfun(@(f) sprintf('%.2f', T.(f)(k)), flds, 'UniformOutput', false);
    end
    body = strrep(tags{b}, '-', ', ');
    body = regexprep(body, '(SR|MR)(\d)', '$1$_$2$');
    lines{end+1} = sprintf('%s & %s\\\\', body, strjoin(vals, ' & ')); %#ok<SAGROW>
  end
  if p < numel(pts), lines{end+1} = '\midrule'; end %#ok<SAGROW>
end

fid = fopen(outFile, 'w');
fprintf(fid, '%s\n', lines{:});
fclose(fid);
fprintf('%s\n', lines{:});
fprintf('\nwrote %s\n', outFile);
end
