cvR = 0.7;
cvS = 0.1;
cvA = -0.9;

shade = linspace(0.1,0.9,3);

cmap1 = jet;
nColors = size(cmap1,1);
val2idx = @(v) round((v - (-1)) / (1 - (-1)) * (nColors - 1)) + 1;

crS = cmap1(val2idx(shade(1)*cvR),:);
crE = cmap1(val2idx(shade(2)*cvR),:);
crB = cmap1(val2idx(shade(3)*cvR),:);

n = 256;
t = linspace(0, 1, n)';

cmap_rS = (1-t) .* [1 1 1] + t .* crS;
cmap_rE = (1-t) .* [1 1 1] + t .* crE;
cmap_rB = (1-t) .* [1 1 1] + t .* crB;

% Nominal model, nominal body
root = '../experiments/hebbian/logs/hebbian-sweeps/';
path0 = [root, 'm0-b0-sweeps_S1_E05_B025', ...
        '/run1/analysis/summary/summary_table.mat'];

ex0 = [root, 'm0-b0-sweeps_S1_E05_B025/run1/', ...
      '1772404269'];



path1 = [root, 'm0-b1-sweeps_S1_E05_B025-P1_5D1_5', ...
        '/run1/analysis/summary/summary_table.mat'];

ex1 = [root, 'm0-b1-sweeps_S1_E05_B025-P1_5D1_5/run1/', ...
      '1773069062'];



path2 = [root, 'm1-b0-sweeps_S1_E025_B05', ...
        '/run1/analysis/summary/summary_table.mat'];

ex2 = [root, 'm1-b0-sweeps_S1_E025_B05/run1/', ...
      '1772894203'];


path3 = [root, 'm1-b1-sweeps_S1_E025_B05-P1_5D1_5', ...
        '/run1/analysis/summary/summary_table.mat'];
ex3 = [root, 'm1-b1-sweeps_S1_E025_B05-P1_5D1_5/run1/', ...
      '1773044089'];


path4 = [root, 'm2-b0-sweeps_S05_E05_B075', ...
        '/run1/analysis/summary/summary_table.mat'];

ex4 = [root, 'm2-b0-sweeps_S05_E05_B075/run1/', ...
      '1772894302'];


path5 = [root, 'm2-b1-sweeps_S05_E05_B075-P1_5D1_5', ...
        '/run1/analysis/summary/summary_table.mat'];


ex5 = [root, 'm2-b1-sweeps_S05_E05_B075-P1_5D1_5/run1/', ...
      '1773044069'];

paths = {path0,path1,path2,path3,path4,path5};
labels = {{'SR_0','MR_0'},{'SR_0','MR_1'}, ...
          {'SR_1','MR_0'},{'SR_1', 'MR_1'}, ...
          {'SR_2','MR_0'},{'SR_2', 'MR_1'}};
examples = {ex0,ex1,ex2,ex3,ex4,ex5};

n = numel(paths);

tab = {};
for i=1:n
  load(paths{i});
  tab{i} = summary_table;
end

nrows = n;
ncols = 9;

xl = [0.1 1];
yl = [0.0003    0.0100];
zl = [0 1];

figure('Position', [431 28  620 950]);

ysp = 0.01;
xsp = 0.01;
xpad = 0.1;
ypad = 0.07;

h = (1 - 2*ypad - (nrows-1)*ysp) / nrows;
w = (0.66 - 2*xpad - (ncols-1)*xsp) / ncols;

xs = xpad;
ys = 1 - ypad - h;

for i=1:n
  iq = find(tab{i}.('heb_kwa_gai') == 3.5);
  x = tab{i}.opt_max_mus_str(iq);
  y = tab{i}.sup_kwa_bab_max_pro(iq);

  Z = [ tab{i}.rS_mean(iq), ...
        tab{i}.rE_mean(iq), ...
        tab{i}.rB_mean(iq), ...
        tab{i}.sS_mean(iq), ...
        tab{i}.sE_mean(iq), ...
        tab{i}.sB_mean(iq), ...
        tab{i}.aS_mean(iq), ...
        tab{i}.aE_mean(iq), ...
        tab{i}.aB_mean(iq)];

  for b=1:size(Z,2)
    xtl = false;
    ytl = false;
    if i == n
      xtl = true;
      ytl = true;
    end

    pos = [xs, ys, w, h];
    ax = axes('Position', pos);

    plot_heatmap(ax,x,y,Z(:,b),...
      xl=xl,yl=yl,zl=zl,scale='log',XTickLabels=xtl,YTickLabels=ytl);

    %if i==1
    %  ann = annotation('textbox','Position',pos+[0,0.02,0,0]);
    %  ann.String = 'Reflex';
    %  ann.EdgeColor = 'none';
    %  ann.HorizontalAlignment = 'center';
    %  ann.VerticalAlignment = 'top';
    %end

    %ann = annotation('textbox','Position',pos+[-xpad,0.0,0,0]);
    %ann.String = labels{i};
    %ann.EdgeColor = 'none';
    %ann.HorizontalAlignment = 'left';
    %ann.VerticalAlignment = 'middle';

    xs = xs + xsp + w;
  end

  %if i==1
  %  ann = annotation('textbox','Position',pos+[0,0.02,0,0]);
  %  ann.String = 'R*S*(1 - A)';
  %  ann.EdgeColor = 'none';
  %  ann.HorizontalAlignment = 'center';
  %  ann.VerticalAlignment = 'top';
  %end

  xs = xpad;
  ys = ys - ysp - h;
end

nrows = 1;
ncols = 4;
w = 0.33-xpad/2;

h = h - 2*ysp;
xs = 0.66;
ys = 1 - ypad -h;

m = numel(examples);
for i=1:m
  fprintf(['Extracting ',examples{i},'\n']);
  % Extract parameters
  [keys, vals] = extract_parameters(examples{i});

  % Extract data
  [episode, time, muscle_strength, Wi, We] = extract_data([examples{i},'/data.txt']);
  motif = ds_Ia_motif(We);

  plot_W_evolution(We,motif, ...
    ys=ys,xs=xs,nhm=11,w=w,h=h,hmsp=0.005)

  ys = ys - h - 3*ysp;
  end

%muscle_strengths = unique(tab{i}.opt_max_mus_str);
%babble_prob = unique(tab{i}.sup_kwa_bab_max_pro);








