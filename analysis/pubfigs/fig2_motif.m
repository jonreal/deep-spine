clear all;

LW = 1;
BW = 0.55; 


fwidth = 910;
fheight = 950;   % taller: two trace rows (B-D, E-G). Tune aspect manually.

%% Layout parameters — edit here
xmargin   = 0.075;
ymargin   = 0.05;


hmalpha = 0.5;

% Heights
h_leg     = 0.10;
h_icon    = 0.10;
leg_h     = 0.070;         % legend-tile axes HEIGHT: keeps tiles height-limited
                           % (~square) at tall figures (width stays h_leg/h_icon)
h_trace   = 0.06;          % per-trace height (two rows; fills the compacted band)
h_bar     = h_trace * 0.75;
w_bar_x   = 0.05;

% Trace-row blocks (B-D = row 1, E-G = row 2). Each block's reflex baseline is
% DERIVED below (yref_of); blocks stack downward and are auto-spaced to fill the
% band between the separator and the bottom margin (ysp_block is derived).
gap_top       = 0.015;     % gap: separator -> top of first row block
y_bot         = 0.04;      % bottom margin: last block's final bar bottoms out here
yhm_shift     = 0.01;
y_final_shift = 0.01;

% Per-column (lambda_max, p_b) header, placed just above each block's heatmaps
hdr_h   = 0.022;           % header textbox height (reserved at the top of a block)
hdr_gap = 0.006;           % gap: heatmap-tile top -> header

% Column layout
ncol      = 3;
xsp_col   = 0.15;

% Heatmap tiles
nhm       = 5;
hmsp      = 0.005;
hm_overhang = 0.045;

% Spacings
ysp       = 0.012;
ysp_sep   = 0.02;

% Legend annotation
icon_v    = 0;
ann_shift = [-0.02, 0, 0.10, 0];
fontsize  = 10;
xbasissp  = -0.001;
gap_top_leg = 0.015;        % space above the legend (figure top -> legend top)

% --- Derived (do not edit below) ---
w      = (1 - 2*xmargin - (ncol-1)*xsp_col) / ncol;
xhm    = (w + xmargin + hm_overhang - (nhm+1)*hmsp) / nhm;
w_cbar = xhm / 6;

% Panel-A: legend pinned to the top; separator sits just below it
y_leg_bot = 1 - gap_top_leg - leg_h;   % legend bottom (also used to place legend)
y_sep     = y_leg_bot - ysp_sep;       % separator just under the legend

% One trace-block footprint, relative to its reflex baseline (yref), top->bottom:
%   header | heatmap tiles | reflex | synergy | anti-synergy | final bar
hm_top   = h_trace + ysp + yhm_shift + xhm;     % yref -> top of heatmap tiles
up_off   = hm_top + hdr_gap + hdr_h;            % yref -> top of block (above header)
down_off = 3*h_trace + 4*ysp + y_final_shift;   % yref -> bottom of final bar
row_h    = up_off + down_off;                   % full vertical footprint / block
% (yref_of and the auto-derived inter-block gap ysp_block are set after nblock)

% ---------------------------------------------------
% Nominal model -- diag organization
% ---------------------------------------------------
root = '../experiments/hebbian/logs/hebbian-sweep/';
ex0 = [root, 'SR0-MR0-gain/', ...
      '1787087457'];

% Extract data
[episode, time, muscle_strength, Wi, We] = extract_data([ex0,'/data.txt']);
motif = ds_Ia_motif(We);
data1 = {motif,We};

[keys, vals] = extract_parameters(ex0);
gain1 = str2num(cell2mat(vals(strcmp(keys,'heb_kwa_gai') == 1)));
str1 = str2num(cell2mat(vals(strcmp(keys,'opt_max_mus_str') == 1)));
prob1 = str2num(cell2mat(vals(strcmp(keys,'sup_kwa_bab_max_pro') == 1)));

% ---------------------------------------------------
% Nominal model -- synergist organization
% ---------------------------------------------------
root = '../experiments/hebbian/logs/hebbian-sweep/';
ex0 = [root, 'SR0-MR0-gain/', ...
      '1787087482'];
% Extract data
[episode, time, muscle_strength, Wi, We] = extract_data([ex0,'/data.txt']);
motif = ds_Ia_motif(We);

[keys, vals] = extract_parameters(ex0);
gain2 = str2num(cell2mat(vals(strcmp(keys,'heb_kwa_gai') == 1)));
str2 = str2num(cell2mat(vals(strcmp(keys,'opt_max_mus_str') == 1)));
prob2 = str2num(cell2mat(vals(strcmp(keys,'sup_kwa_bab_max_pro') == 1)));

data2 = {motif,We};

% ---------------------------------------------------
% Nominal model -- maladaptive organization
% ---------------------------------------------------
root = '../experiments/hebbian/logs/hebbian-sweep/';
ex0 = [root, 'SR0-MR0-gain/', ...
      '1787087512'];
% Extract data
[episode, time, muscle_strength, Wi, We] = extract_data([ex0,'/data.txt']);
motif = ds_Ia_motif(We);

[keys, vals] = extract_parameters(ex0);
gain3 = str2num(cell2mat(vals(strcmp(keys,'heb_kwa_gai') == 1)));
str3 = str2num(cell2mat(vals(strcmp(keys,'opt_max_mus_str') == 1)));
prob3 = str2num(cell2mat(vals(strcmp(keys,'sup_kwa_bab_max_pro') == 1)));

data3 = {motif,We};

% ---------------------------------------------------
Data1 = {data1, data2, data3};
P1 = {{gain1,str1,prob1},{gain1,str2,prob2},{gain3,str3,prob3}};

% ---------------------------------------------------
% Nominal model -- diag organization
% ---------------------------------------------------
root = '../experiments/hebbian/logs/hebbian-sweep/';
ex0 = [root, 'SR0-MR0/', ...
      '1786936211'];
% Extract data
[episode, time, muscle_strength, Wi, We] = extract_data([ex0,'/data.txt']);
motif = ds_Ia_motif(We);
data1 = {motif,We};

[keys, vals] = extract_parameters(ex0);
gain1 = str2num(cell2mat(vals(strcmp(keys,'heb_kwa_gai') == 1)));
str1 = str2num(cell2mat(vals(strcmp(keys,'opt_max_mus_str') == 1)));
prob1 = str2num(cell2mat(vals(strcmp(keys,'sup_kwa_bab_max_pro') == 1)));

% ---------------------------------------------------
% Nominal model -- synergist organization
% ---------------------------------------------------
root = '../experiments/hebbian/logs/hebbian-sweep/';
ex0 = [root, 'SR0-MR0/', ...
      '1786936221'];
% Extract data
[episode, time, muscle_strength, Wi, We] = extract_data([ex0,'/data.txt']);
motif = ds_Ia_motif(We);

[keys, vals] = extract_parameters(ex0);
gain2 = str2num(cell2mat(vals(strcmp(keys,'heb_kwa_gai') == 1)));
str2 = str2num(cell2mat(vals(strcmp(keys,'opt_max_mus_str') == 1)));
prob2 = str2num(cell2mat(vals(strcmp(keys,'sup_kwa_bab_max_pro') == 1)));

data2 = {motif,We};

% ---------------------------------------------------
% Nominal model -- maladaptive organization
% ---------------------------------------------------
root = '../experiments/hebbian/logs/hebbian-sweep/';
ex0 = [root, 'SR0-MR0/', ...
      '1786936250'];

% Extract data
[episode, time, muscle_strength, Wi, We] = extract_data([ex0,'/data.txt']);
motif = ds_Ia_motif(We);

[keys, vals] = extract_parameters(ex0);
gain3 = str2num(cell2mat(vals(strcmp(keys,'heb_kwa_gai') == 1)));
str3 = str2num(cell2mat(vals(strcmp(keys,'opt_max_mus_str') == 1)));
prob3 = str2num(cell2mat(vals(strcmp(keys,'sup_kwa_bab_max_pro') == 1)));

data3 = {motif,We};

% ---------------------------------------------------
Data2 = {data1, data2, data3};
P2 = {{gain1,str1,prob1},{gain2,str2,prob2},{gain3,str3,prob3}};

nmaxs1 = {180000,180000,180000};
nmaxs2 = {180000,180000,180000};

% data.txt is logged at 1 Hz, so sample index = sim seconds; plot in hours
tscale = 1/3600;

% Row-blocks: reuse the same three columns for both rows for now. To use
% different data in row 2 (E-G), replace DataBlocks{2} / Pblocks{2} / nmaxblocks{2}.
DataBlocks = {Data1, Data2};
Pblocks    = {P1,    P2};
nmaxblocks = {nmaxs1, nmaxs2};
nblock     = numel(DataBlocks);

% Auto-space the blocks to fill [y_bot, y_sep-gap_top] (no leftover at bottom).
if nblock > 1
  ysp_block = (y_sep - gap_top - y_bot - nblock*row_h) / (nblock - 1);
else
  ysp_block = 0;
end
% reflex baseline of block rb (blocks stack downward below the separator)
yref_of = @(rb) (y_sep - gap_top - up_off) - (rb-1)*(row_h + ysp_block);

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
n = 2048;
blue = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
red  = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
cmap2 = [blue; red];
cmap2 = [red];
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
xs_fudge = 0.035
ys = y_leg_bot;   % legend pinned to the top (see y_leg_bot in derived section)

shift = ann_shift;

pos = [xs-xs_fudge, ys, h_leg, leg_h];
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
ax.XAxis.FontSize = 8;
ax.YAxis.FontSize = 8;
xlabel('Ia Neurons','fontsize',10);
ylabel('Motoneurons','fontsize',10);
ax.TickDir = 'out';
box off;


xs = xs-xs_fudge + h_leg + xbasissp;
for i=1:3
  pos = [xs, ys+h_icon*icon_v, h_icon, leg_h];
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

  xs = xs + h_icon + xbasissp;

end


for i=1:3
  pos = [xs, ys+h_icon*icon_v, h_icon, leg_h];
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

  xs = xs + h_icon + xbasissp;
end
for i=1:3
  pos = [xs, ys+h_icon*icon_v, h_icon, leg_h];
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

  xs = xs + h_icon + xbasissp;
end

ann = annotation('line', [0.01, 0.99], [y_sep-0.01, y_sep-0.01]);
ann.Color = [0.75, 0.75, 0.75];


% -----------------------------------------------------------------------------
% Traces  (two row-blocks: B-D = block 1, E-G = block 2)
% -----------------------------------------------------------------------------
xsp = xsp_col;

for rb=1:nblock

  Data  = DataBlocks{rb};
  P     = Pblocks{rb};
  nmaxs = nmaxblocks{rb};

  % reflex baseline + within-block anchors for this row
  ys_ref  = yref_of(rb);
  ys_hm   = ys_ref + h_trace + ysp;
  ys_syn  = ys_ref - h_trace - ysp;
  ys_anti = ys_syn - h_trace - ysp;
  ys_bar  = ys_anti - h_trace - 2*ysp - y_final_shift;

  xs = xmargin;
  h  = h_trace;
  ys = ys_ref;

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
  xl = [0 nmax]*tscale;
  %s = scatter(r,motif.r(:,1),markerSize,'.'); hold on;
  %s.MarkerEdgeColor = shade(1)*cR;
  %s = scatter(r,motif.r(:,2),markerSize,'.');
  %s.MarkerEdgeColor = shade(2)*cR;
  %s = scatter(r,motif.r(:,3),markerSize,'.');
  %s.MarkerEdgeColor = shade(3)*cR;

  for k=1:3
    plot(r*tscale,motif.r(:,k), ...
        Color=shade(k)*cR, ...
        LineWidth=LW);
    hold on;
  end

  ylim([0,1]);
  yticklabels({0,'',1});
  xlim(xl);
  xticks(linspace(0,xl(2),nhm));   % one tick per W snapshot
  if i==1
    ylabel('Reflex')
    xticklabels({});
  else
    yticklabels({});
    xticklabels({});
  end
  ax.Color = 'none';

  ax_reflex = ax;

  ann = annotation('textbox', [xs-0.02, ys_ref + hm_top + hdr_gap, w+0.05, hdr_h]);
  ann.String = ['\kappa = ',num2str(P{i}{1}), ...
                ',    \lambda_{max} = ',num2str(P{i}{2}), ...
                ',    p_b = ', pb_str(P{i}{3})];
  ann.EdgeColor = 'none';
  ann.HorizontalAlignment = 'center';
  ann.VerticalAlignment = 'bottom';
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
    plot(r*tscale,motif.sp(:,k), ...
        Color=shade(k)*cS, ...
        LineWidth=LW);
    hold on;
  end


  ylim([0,1]);
  yticklabels({0,'',1});
  xlim(xl);
  xticks(linspace(0,xl(2),nhm));   % one tick per W snapshot
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
    plot(r*tscale,motif.ap(:,k), ...
        Color=shade(k)*cA, ...
        LineWidth=LW);
    hold on;
  end

  ylim([0,1]);
  yticklabels({0,'',1});
  xlim(xl);
  xticks(linspace(0,xl(2),nhm));   % one tick per W snapshot
  if i==1
    ylabel('Antagonist')
    xlabel('Time (h)')
  else
    xlabel('Time (h)')
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
    x_trace = ax_pos(1) + ax_pos(3) * (indx*tscale - xl(1)) / (xl(2) - xl(1));
    y_trace = ax_pos(2) + ax_pos(4);     % top edge

    ann = annotation('line', [x_hm, x_trace], [y_hm, y_trace]);
    ann.Color = [0.8 0.8 0.8 0.1];
    ann.LineWidth = 0.1;

    xs = xs + xhm + hmsp;
  end

  if i==3 && rb==1
    pos_leg = [pos(1)+xhm*0.66, ys_hm-0.065, w_cbar, xhm];
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

  xs = xmargin;
  xs = xs + i*(w + xsp);
  ys = ys_ref;

end   % columns

end   % row-blocks

% horizontal divider(s) between row blocks (match the panel-A legend separator)
for rb=1:(nblock-1)
  y_line = (yref_of(rb) - down_off + yref_of(rb+1) + up_off - 0.01)  / 2;  % center of gap
  ann = annotation('line', [0.01, 0.99], [y_line, y_line]);
  ann.Color = [0.75, 0.75, 0.75];
end

% vertical dividers between columns, drawn PER block (one row each) so they
% stop at the block edges, leaving a small gap around the horizontal dividers
% -- the same partition look as the legend break vs. the column breaks.
for rb=1:nblock
  y_top    = yref_of(rb) + up_off;     % block top (header)
  y_bottom = yref_of(rb) - down_off - 0.01;   % block bottom (final bar)
  for i=1:(ncol-1)
    x_div = xmargin + i*(w + xsp_col) - xsp_col/2;
    ann = annotation('line', [x_div, x_div], [y_bottom, y_top]);
    ann.Color = [0.75, 0.75, 0.75];
  end
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

% p_b as tex scientific notation with 2 significant figures, e.g. 4.6\times10^{-3}
function s = pb_str(p)
  e = floor(log10(p));
  m = round(p/10^e*10 + 1e-9)/10;   % nudge: 0.0215 is stored as 2.1499...e-2
  s = sprintf('%.1f\\times10^{%d}', m, e);
end
