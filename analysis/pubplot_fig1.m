clear all;

LW = 1;
BW = 0.75; 


fwidth = 910;
fheight = 520;

%% Layout parameters — edit here
xmargin   = 0.075;
ymargin   = 0.05;


hmalpha = 0.5;

% Heights
h_leg     = 0.10;
h_icon    = 0.10;
h_trace   = 0.12;
h_bar     = h_trace * 0.75;
w_bar_x   = 0.05;

% Start of the traces
ys_ref  = 0.52;
yhm_shift = 0.02;
y_final_shift = 0.02;

% Column layout
ncol      = 3;
xsp_col   = 0.15;

% Heatmap tiles
nhm       = 5;
hmsp      = -0.03;
hm_overhang = 0.025;

% Spacings
ysp       = 0.02;
ysp_sep   = 0.02;

% Legend annotation
icon_v    = 0;
ann_shift = [-0.02, 0, 0.10, 0];
fontsize  = 9;
xbasissp  = -0.005;

% --- Derived (do not edit below) ---
w      = (1 - 2*xmargin - (ncol-1)*xsp_col) / ncol;
xhm    = (w + xmargin + hm_overhang - (nhm+1)*hmsp) / nhm;
w_cbar = xhm / 6;

% Vertical stack, top → bottom
ys_leg  = 1 - ymargin - h_leg;
y_sep   = ys_leg - ysp_sep;
ys_hm   = ys_ref + h_trace + ysp;
ys_syn  = ys_ref  - h_trace - ysp;
ys_anti = ys_syn  - h_trace - ysp;
ys_bar  = ys_anti - h_trace - 2*ysp - y_final_shift;

% ---------------------------------------------------
% Nominal model -- diag organization
% ---------------------------------------------------
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

% ---------------------------------------------------
% Nominal model -- synergist organization
% ---------------------------------------------------
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

% ---------------------------------------------------
% Nominal model -- maladaptive organization
% ---------------------------------------------------
root = '../experiments/hebbian/logs/hebbian-sweeps/';
ex0 = [root, 'm0-b0-sweeps_S1_E05_B025/run1/', ...
      '1772512202'];
% Extract data
[episode, time, muscle_strength, Wi, We] = extract_data([ex0,'/data.txt']);
motif = ds_Ia_motif(We);

[keys, vals] = extract_parameters(ex0);
str3 = str2num(cell2mat(vals(strcmp(keys,'opt_max_mus_str') == 1)));
prob3 = str2num(cell2mat(vals(strcmp(keys,'sup_kwa_bab_max_pro') == 1)));

data3 = {motif,We};

% ---------------------------------------------------
Data = {data1, data2, data3};
P = {{str1,prob1},{str2,prob2},{str3,prob3}};

nmaxs = {10000,100000,100000};

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
n = 1024;
blue = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
red  = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
cmap2 = [blue; red];
%cmap2 = flipud(brewermap(1024, 'RdBu'));
%cmap2 = pink(1024);

% Figure
figure('position',[1440 818 fwidth fheight]);

xmargin = 0.075;
ymargin = 0.1;


%xs = xmargin/2;
%ys = 1 - ymargin/2 - 1.5*h;

% -----------------------------------------------------------------------------
% Legend
% -----------------------------------------------------------------------------
lab = {'S','E','B'};
shade = linspace(0.5,1,3);

xs = 0.05;
ys = ys_leg + ymargin/4;

shift = ann_shift;

pos = [xs, ys, h_leg, h_leg];
ax = axes('position',pos);

rgb = motifs_to_rgb(motif, cR, cS, cA, shade);
p = image(rgb);
p.AlphaData = double(~isnan(rgb(:,:,1))) * 0.8;
axis square;
%ax.XAxisLocation = 'top';
yticks(1:6);
xticks(1:6);
neuron_lab = {'SF', 'SE', 'EF', 'EE', 'BF', 'BE'};
yticklabels(neuron_lab);
xticklabels(neuron_lab);
ax.XTickLabelRotation = 0;
ax.XAxis.FontSize = 6;
ax.YAxis.FontSize = 6;
xlabel('Ia Neurons','fontsize',8);
ylabel('Motoneurons','fontsize',8);
ax.TickDir = 'out';
box off;


xs = xs + h_leg + xbasissp;
for i=1:3
  pos = [xs, ys+h_icon*icon_v, h_icon, h_icon];
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
    ann.String = [' =    r_',lab{i}];
  else
    ann.String = [' +    r_',lab{i}];
  end
  ann.EdgeColor ='none';
  ann.HorizontalAlignment = 'left';
  ann.VerticalAlignment = 'middle';
  ann.FontSize = fontsize;

  xs = xs + h_icon + xbasissp;

end


for i=1:3
  pos = [xs, ys+h_icon*icon_v, h_icon, h_icon];
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
  ann.String = [' +   s_',lab{i}];
  ann.EdgeColor ='none';
  ann.HorizontalAlignment = 'left';
  ann.VerticalAlignment = 'middle';
  ann.FontSize = fontsize;

  xs = xs + h_icon + xbasissp;
end
for i=1:3
  pos = [xs, ys+h_icon*icon_v, h_icon, h_icon];
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
  ann.String = [' +   a_',lab{i}];
  ann.EdgeColor ='none';
  ann.HorizontalAlignment = 'left';
  ann.VerticalAlignment = 'middle';
  ann.FontSize = fontsize;

  xs = xs + h_icon + xbasissp;
end

ann = annotation('line', [0.01, 0.99], [y_sep, y_sep]);
ann.Color = [0.75, 0.75, 0.75];


% -----------------------------------------------------------------------------
% Traces
% -----------------------------------------------------------------------------
xsp = xsp_col;
xs  = xmargin;
h   = h_trace;
ys  = ys_ref;


for i=1:ncol

  dq = Data{i};
  motif = dq{1};
  We = dq{2};
  nmax = nmaxs{i};

  % ---- Reflex
  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  n = numel(motif.rg);
  r = 1:n;
  xl = [0 nmax];
  %s = scatter(r,motif.r(:,1),markerSize,'.'); hold on;
  %s.MarkerEdgeColor = shade(1)*cR;
  %s = scatter(r,motif.r(:,2),markerSize,'.');
  %s.MarkerEdgeColor = shade(2)*cR;
  %s = scatter(r,motif.r(:,3),markerSize,'.');
  %s.MarkerEdgeColor = shade(3)*cR;

  for k=1:3
    plot(r,motif.r(:,k), ...
        Color=shade(k)*cR, ...
        LineWidth=LW);
    hold on;
  end

  ylim([0,1]);
  yticklabels({0,'',1});
  xlim(xl);
  if i==1
    ylabel('Reflex')
    xticklabels({});
  else
    yticklabels({});
    xticklabels({});
  end
  ax.Color = 'none';

  ax_reflex = ax;

  ann = annotation('textbox', pos+[-0.05, 3*h_trace/4 + 0.02, 0.1, 0.075]);
  ann.String = ['\lambda_{max} = ',num2str(P{i}{1}), ...
                ',    p_b = ', num2str(P{i}{2},'%.e')];
  ann.EdgeColor = 'none';
  ann.HorizontalAlignment = 'center';
  ann.Interpreter = 'tex';

  ys = ys_syn;

  % ---- Synergy
  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  %s = scatter(r,motif.sp(:,1),markerSize,'.'); hold on;
  %s.MarkerEdgeColor = shade(1)*cS;
  %s = scatter(r,motif.sp(:,2),markerSize,'.');
  %s.MarkerEdgeColor = shade(2)*cS;
  %s = scatter(r,motif.sp(:,3),markerSize,'.');
  %s.MarkerEdgeColor = shade(3)*cS;

  for k=1:3
    plot(r,motif.sp(:,k), ...
        Color=shade(k)*cS, ...
        LineWidth=LW);
    hold on;
  end


  ylim([0,1]);
  yticklabels({0,'',1});
  xlim(xl);
  if i==1
    ylabel('Synergy')
    xticklabels({});
  else
    yticklabels({});
    xticklabels({});
  end
  ax.Color = 'none';

  ys = ys_anti;

  % ---- Anti-Synergy
  pos = [xs,ys,w,h];
  ax = axes('position',pos);

%  s = scatter(r,motif.ap(:,1),markerSize,'.'); hold on;
%  s.MarkerEdgeColor = shade(1)*cA;
%  s = scatter(r,motif.ap(:,2),markerSize,'.');
%  s.MarkerEdgeColor = shade(2)*cA;
%  s = scatter(r,motif.ap(:,3),markerSize,'.');
%  s.MarkerEdgeColor = shade(3)*cA;

  for k=1:3
    plot(r,motif.ap(:,k), ...
        Color=shade(k)*cA, ...
        LineWidth=LW);
    hold on;
  end

  ylim([0,1]);
  yticklabels({0,'',1});
  xlim(xl);
  xticks(linspace(0,nmaxs{i},3));
  if i==1
    ylabel('Anti-Synergy')
    xlabel('Sim steps')
  else
    xlabel('Sim steps')
    yticklabels({});
  end
  ax.Color = 'none';

  D = [mean(motif.rp(nmax-200:nmax,:),1), ...
        mean(motif.sp(nmax-200:nmax,:),1), ...
        mean(motif.ap(nmax-200:nmax,:),1)]';

  C = [shade(1).*cR; shade(2).*cR; shade(3).*cR; ...
        shade(1).*cS; shade(2).*cS; shade(3).*cS; ...
        shade(1).*cA; shade(2).*cA; shade(3).*cA];


  % ---- Final
  pos = [xs-w_bar_x/2, ys_bar, w+w_bar_x, h_bar];
  ax = axes('Position',pos);
  b = bar(D, 'LineWidth', 0.25, 'FaceColor', 'flat');
  b.BarWidth = BW;
  b.CData = C;
  b.EdgeColor = 'none';
  grid on;
    ylim([0,1]);
    yticks(linspace(0,1,5));
    yticklabels({0, '', '', '', 1});
    xticklabels({'r_S', 'r_E', 'r_B', 's_S', 's_E', 's_B', 'a_S', 'a_E', 'a_B'});
    xtickangle(0);
  ax.Color = 'none';
  if i==1
    ylabel('Final Weights')
  end

  %----- Weight Transformation
  xs = xs - xmargin/2 - hm_overhang/2 + hmsp;

  hm_indices = round(linspace(0, min(nmax, numel(We)-1), nhm));
  for j=1:nhm
    pos = [xs, ys_hm+yhm_shift, xhm, xhm];
    ax = axes('pos',pos);
    indx = hm_indices(j);
    p = imagesc(We{indx+1});
    p.AlphaData = hmalpha;
    colormap(ax, cmap2);
    caxis([0,1]);
    axis square;
    axis off;

    x_hm = pos(1) + pos(3)/2;
    y_hm = pos(2);

    ax_pos = ax_reflex.Position;          % [x y w h] in figure
    xl = ax_reflex.XLim;
    x_trace = ax_pos(1) + ax_pos(3) * (indx - xl(1)) / (xl(2) - xl(1));
    y_trace = ax_pos(2) + ax_pos(4);     % top edge

    ann = annotation('line', [x_hm, x_trace], [y_hm, y_trace]);
    ann.Color = [0.8 0.8 0.8 0.1];
    ann.LineWidth = 0.1;

    xs = xs + xhm + hmsp;
  end

  if i==3
    pos_leg = [pos(1)+xhm/2, ys_hm-0.1, w_cbar, xhm];
    ax = axes('pos',pos_leg);
    p = imagesc([0 1], [0 1], flipud(linspace(0,1,1024)'));
    p.AlphaData = hmalpha;
    ylim([0,1]);
    ax.XAxis.Visible = 'off';
    ax.YAxisLocation = 'right';
    yticklabels({'1', '', '0'});
    colormap(ax,cmap2);
    ax.Color = 'none';
    caxis([0,1]);
  end

  if (i==1) || (i==2)
    x_div = xmargin + i*(w + xsp) - xsp/2;
    x = [x_div, x_div];
    y = [0.01, 0.82];
    ann = annotation('line',x,y);
    ann.Color = [0.75, 0.75, 0.75];
  end


  xs = xmargin;
  xs = xs + i*(w + xsp);
  ys = ys_ref;

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
