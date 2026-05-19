clear all;
% Nominal model, nominal body
root = '../experiments/hebbian/logs/hebbian-sweeps/';
ex0 = [root, 'm0-b0-sweeps_S1_E05_B025/run1/', ...
      '1772404285'];
% Extract data
[episode, time, muscle_strength, Wi, We] = extract_data([ex0,'/data.txt']);
motif = ds_Ia_motif(We);
data1 = {motif,We};

[keys, vals] = extract_parameters(ex0);
str1 = str2num(cell2mat(vals(strcmp(keys,'opt_max_mus_str') == 1)));
prob1 = str2num(cell2mat(vals(strcmp(keys,'sup_kwa_bab_max_pro') == 1)));

root = '../experiments/hebbian/logs/hebbian-sweeps/';
ex0 = [root, 'm0-b0-sweeps_S1_E05_B025/run1/', ...
      '1772485658'];
% Extract data
[episode, time, muscle_strength, Wi, We] = extract_data([ex0,'/data.txt']);
motif = ds_Ia_motif(We);

[keys, vals] = extract_parameters(ex0);
str2 = str2num(cell2mat(vals(strcmp(keys,'opt_max_mus_str') == 1)));
prob2 = str2num(cell2mat(vals(strcmp(keys,'sup_kwa_bab_max_pro') == 1)));

data2 = {motif,We};
Data = {data1, data2, data2};
P = {{str1,prob1},{str2,prob2},{str2,prob2}};

nmax = 100000;

% Colors
cvR = 0.6;
cvS = 0.0;
cvA = -0.9;
cmap1 = parula;
nColors = size(cmap1,1);
val2idx = @(v) round((v - (-1)) / (1 - (-1)) * (nColors - 1)) + 1;
cR = cmap1(val2idx(cvR),:);
cS = cmap1(val2idx(cvS),:);
cA = cmap1(val2idx(cvA),:);

% Color for weights
n = 64;
blue = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
red  = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
cmap2 = [blue; red];
cmap2 = flipud(brewermap(1024, 'RdBu'));

% Figure
figure('position',[1440 818 360 520]);

xmargin = 0.075;
ymargin = 0.1;

xsp = 0.01;
ysp = 0.02;

nhm = 5;
hmsp = 0.002;
hmalpha = 1;

%xs = xmargin/2;
%ys = 1 - ymargin/2 - 1.5*h;

% -----------------------------------------------------------------------------
% Legend
% -----------------------------------------------------------------------------
lab = {'S','E','B'};
shade = linspace(0.5,1,3);

xs = 0.05;
ys = 0.85;
h = 0.1;
w = 0.1;

xbasissp = 0.03;
shift = [-0.04,0,0.1,0];
fontsize = 9;

pos = [xs,ys,w,h];
ax = axes('position',pos);

rgb = motifs_to_rgb(motif, cR, cS, cA, shade);
p = image(rgb);
p.AlphaData = double(~isnan(rgb(:,:,1))) * 0.8;
axis square;
ax.XAxisLocation = 'top';
yticks(1:6);
xticks(1:6);
yticklabels({'MN-SF', 'MN-SE', 'MN-EF', 'MN-EE', 'MN-BF', 'MN-BE'});
xticklabels({'Ia-SF', 'Ia-SE', 'Ia-EF', 'Ia-EE', 'Ia-BF', 'Ia-BE'});
ax.XTickLabelRotation = 45;
ax.XAxis.FontSize = 7;
ax.YAxis.FontSize = 7;

xs = xs + w + xbasissp;
h = 0.06;
w = 0.06;
for i=1:3
  pos = [xs,ys+h/4,w,h];
  ax = axes('position',pos);

  Rmi = reshape(motif.maskp(:,i),6,6);

  rgb = mask_to_rgb(Rmi, cR, shade(i));

  p = image(rgb);
  p.AlphaData = double(Rmi ~= 0);
  grid on;
  box off;
  ax.Color = 'none';
  ticks = linspace(0.5,6.5,7);
  xticks(ticks);
  yticks(ticks);
  %yticklabels({});
  %xticklabels({});
  caxis([-1,1]);
  axis square;
  ax.XAxis.Visible = 'off';
  ax.YAxis.Visible = 'off';


  ann = annotation('textbox',pos+shift);
  if i==1
    ann.String = [' = r_',lab{i}];
  else
    ann.String = [' + r_',lab{i}];
  end
  ann.EdgeColor ='none';
  ann.HorizontalAlignment = 'left';
  ann.VerticalAlignment = 'middle';
  ann.FontSize = fontsize;

  xs = xs + w + xbasissp;

end


for i=1:3
  pos = [xs,ys+h/8,w,h];
  ax = axes('position',pos);

  Smi = reshape(motif.maskp(:,i+3),6,6);

  rgb = mask_to_rgb(Smi, cS, shade(i));
  p = image(rgb);
  p.AlphaData = double(Smi ~= 0);

  grid on;
  box off;
  ax.Color = 'none';
  ticks = linspace(0.5,6.5,7);
  xticks(ticks);
  yticks(ticks);
  caxis([-1,1]);
  axis square;
  ax.XAxis.Visible = 'off';
  ax.YAxis.Visible = 'off';

  ann = annotation('textbox',pos+shift);
  ann.String = [' + s_',lab{i}];
  ann.EdgeColor ='none';
  ann.HorizontalAlignment = 'left';
  ann.VerticalAlignment = 'middle';
  ann.FontSize = fontsize;

  xs = xs + w + xbasissp;
end
for i=1:3
  pos = [xs,ys+h/8,w,h];
  ax = axes('position',pos);

  Ami = reshape(motif.maskp(:,i+6),6,6);

  rgb = mask_to_rgb(Ami, cA, shade(i));
  p = image(rgb);
  p.AlphaData = double(Ami ~= 0);

  grid on;
  box off;
  ax.Color = 'none';
  ticks = linspace(0.5,6.5,7);
  xticks(ticks);
  yticks(ticks);
  caxis([-1,1]);
  axis square;
  ax.XAxis.Visible = 'off';
  ax.YAxis.Visible = 'off';

  ann = annotation('textbox',pos+shift);
  ann.String = [' + a_',lab{i}];
  ann.EdgeColor ='none';
  ann.HorizontalAlignment = 'left';
  ann.VerticalAlignment = 'middle';
  ann.FontSize = fontsize;

  xs = xs + w + xbasissp;
end



% -----------------------------------------------------------------------------
% Traces
% -----------------------------------------------------------------------------
xsp = 0.15;
ncol = 3;
xs = xmargin;
ys = ys - 3*h - ysp;
h = 0.15;
w = (1 - 2*xmargin - (ncol-1)*xsp)/ncol;

ys = 0.57;

for i=1:ncol

  dq = Data{i};
  motif = dq{1};
  We = dq{2};

  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  n = numel(motif.rg);
  r = 1:n;
  xl = [0 nmax];
  s = scatter(r,motif.r(:,1),0.01,'.'); hold on;
  s.MarkerEdgeColor = shade(1)*cR;
  s = scatter(r,motif.r(:,2),0.01,'.');
  s.MarkerEdgeColor = shade(2)*cR;
  s = scatter(r,motif.r(:,3),0.01,'.');
  s.MarkerEdgeColor = shade(3)*cR;

  %s = plot(r,motif.rg,'linewidth',0.1);
  %s.Color = [1 0 0 0.5];

  ylim([0,1]);
  xlim(xl);
  if i==1
    ylabel('Reflex')
    xticklabels({});
  else
    yticklabels({});
    xticklabels({});
  end
  ax.Color = 'none';

  ann = annotation('textbox',pos+[0,3*h/4,0,0]);
  ann.String = ['\gamma = ',num2str(P{i}{1}), ...
                ',    p_b = ', num2str(P{i}{2},'%.e')];
  ann.EdgeColor = 'none';
  ann.HorizontalAlignment = 'center';

  ys = 1 - ymargin/2 - 3.5*h - ysp;

  %% Motif Strength (Additive)
  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  s = scatter(r,motif.sp(:,1),0.1,'.'); hold on;
  s.MarkerEdgeColor = shade(1)*cS;
  s = scatter(r,motif.sp(:,2),0.1,'.');
  s.MarkerEdgeColor = shade(2)*cS;
  s = scatter(r,motif.sp(:,3),0.1,'.');
  s.MarkerEdgeColor = shade(3)*cS;

  %s = plot(r,motif.sg,'linewidth',0.1);
  %s.Color = 'k';

  ylim([0,1]);
  xlim(xl);
  if i==1
    ylabel('Synergy')
    xticklabels({});
  else
    yticklabels({});
    xticklabels({});
  end
  ax.Color = 'none';

  ys = 1 - ymargin/2 - 4.5*h - 2*ysp;

  %% Motif Strength (Additive)
  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  s = scatter(r,motif.ap(:,1),0.1,'.'); hold on;
  s.MarkerEdgeColor = shade(1)*cA;
  s = scatter(r,motif.ap(:,2),0.1,'.');
  s.MarkerEdgeColor = shade(2)*cA;
  s = scatter(r,motif.ap(:,3),0.1,'.');
  s.MarkerEdgeColor = shade(3)*cA;

  %s = plot(r,motif.ag,'linewidth',0.1);
  %s.Color = 'k';

  ylim([0,1]);
  xlim(xl);
  if i==1
    ylabel('Anti-Synergy')
    xlabel('Simulation steps')
  else
    xlabel('Simulation steps')
    yticklabels({});
  end
  ax.Color = 'none';

  D = [mean(motif.rp(end-200:end,:),1), ...
        mean(motif.sp(end-200:end,:),1), ...
        mean(motif.ap(end-200:end,:),1)]';

  C = [shade(1).*cR; shade(2).*cR; shade(3).*cR; ...
        shade(1).*cS; shade(2).*cS; shade(3).*cS; ...
        shade(1).*cA; shade(2).*cA; shade(3).*cA];


  pos = [xs, ys-h-2*ysp, w, h*0.75];
  ax = axes('Position',pos);
  b = bar(D, 'LineWidth', 0.25, 'FaceColor', 'flat');
  b.CData = C;
  b.EdgeColor = 'none';
  grid on;
    ylim([0,1]);
    yticks(linspace(0,1,5));
    yticklabels({0, '', '', '', 1});
    xticklabels({'r_S', 'r_E', 'r_B', 's_S', 's_E', 's_B', 'a_S', 'a_E', 'a_B'});
  ax.Color = 'none';
  if i==1
    ylabel('Final Weights')
  end

  % Weight Transformation
  xhm = (w + xmargin - (nhm+1)*hmsp) / nhm;
  ys = 1 - ymargin/2 - 2.5*h;

  xs = xs - xmargin/2 + hmsp;
  for j=1:nhm
    pos = [xs,ys + h+ysp,xhm,xhm];
    ax = axes('pos',pos);
    indx = floor((j-1)*numel(r)/nhm);
    p = imagesc(We{indx+1});
    p.AlphaData = hmalpha;
    colormap(ax, cmap2);
    caxis([0,1]);
    axis square;
    axis off;
    xs = xs + xhm + hmsp;
  end

  pos_leg = pos;
  pos_leg = pos_leg + [1.25*xhm +xhm*0.0 0 0];
  pos_leg(3) = xhm/4;
  pos_leg(4) = xhm;
  ax = axes('pos',pos_leg);
  p = imagesc([0 1], [0 1], flipud(linspace(0,1,1024)'));
  p.AlphaData = 0.8;
  ylim([0,1]);
  ax.XAxis.Visible = 'off';
  ax.YAxisLocation = 'right';
  yticklabels({'1', '', '0'});
  colormap(ax,cmap2);
  ax.Color = 'none';
  caxis([0,1]);

  if i==1
    x = [xs+xsp/4, xs+xsp/4];
    y = [0.01, 0.82];
    annotation('line',x,y)
  end


  xs = xmargin;
  xs = xs + i*(w + xsp);
  ys = 0.85;
  ys = ys - 3*h - ysp;
  ys = 0.57

end

% -------------------------------------------------------------------------
function rgb = motifs_to_rgb(motif, cR, cS, cA, shade)
  % Combined RGB: each motif type colored by cR/cS/cA, shaded by level.
  rgb = zeros(6, 6, 3);
  for i = 1:3
    Rmi = reshape(motif.maskp(:,i),   6, 6);
    Smi = reshape(motif.maskp(:,i+3), 6, 6);
    Ami = reshape(motif.maskp(:,i+6), 6, 6);
    for ch = 1:3
      rgb(:,:,ch) = rgb(:,:,ch) ...
        + shade(i) * cR(ch) * double(Rmi) ...
        + shade(i) * cS(ch) * double(Smi) ...
        + shade(i) * cA(ch) * double(Ami);
    end
  end
  any_mask = reshape(any(motif.maskp(:,1:9), 2), 6, 6);
  rgb(repmat(~any_mask, 1, 1, 3)) = NaN;
end

% -------------------------------------------------------------------------
function rgb = mask_to_rgb(mask, color, shade_val)
  dark_c = shade_val * color;
  rgb = zeros(size(mask, 1), size(mask, 2), 3);
  for ch = 1:3
    rgb(:,:,ch) = dark_c(ch) * double(mask);
  end
  rgb(repmat(~mask, 1, 1, 3)) = NaN;
end
