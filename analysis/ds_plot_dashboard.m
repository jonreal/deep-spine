function ds_plot_dashboard( ...
    W,muscle_strength,motif,keys_str,vals_str,name)

  cvR = 0.7;
  cvS = 0.1;
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

  %cmap2 = cbrewer('div','RdGy',256);
  %
  sv_colorval = linspace(-1,1,6);
  cmap3 = copper;
  nColors = size(cmap3,1);
  val2idx = @(v) round((v - (-1)) / (1 - (-1)) * (nColors - 1)) + 1;

  sv_color = cmap3(val2idx(sv_colorval),:);

  figure('position',[1440 818 360 520]);

  ylabsp = 0.02;
  xlabsp = 0.01;
  xmargin = 0.2;
  ymargin = 0.1;

  xsp = 0.01;
  ysp = 0.02;

  nhm = 10;
  hmsp = 0.005;

  % Top Row: Legend, Motif and Rank 1 summary
  %
  ncol = 1;
  xl = [0 numel(W)];

  h = 0.15;
  w = (1 - xmargin - (ncol-1)*xsp)/ncol;

  n = numel(motif.rg);
  r = 1:n;

  xs = xmargin/2;
  ys = 1 - ymargin/2 - 1.5*h;

  %% Motif Strength (Additive)
  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  shade = linspace(0.4,0.8,3);
  s = scatter(r,motif.r(:,1),0.01,'.'); hold on;
  s.MarkerEdgeColor = shade(1)*cR;
  s = scatter(r,motif.r(:,2),0.01,'.');
  s.MarkerEdgeColor = shade(2)*cR;
  s = scatter(r,motif.r(:,3),0.01,'.');
  s.MarkerEdgeColor = shade(3)*cR;

  s = plot(r,motif.rg,'linewidth',0.1);
  s.Color = 'k';

  ylim([0,1]);
  xlim(xl);
  ylabel('Reflex')
  xticklabels({});
  ax.Color = 'none';

  ys = 1 - ymargin/2 - 2.5*h - ysp;

  %% Motif Strength (Additive)
  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  shade = linspace(0.4,0.8,3);
  s = scatter(r,motif.sp(:,1),0.1,'.'); hold on;
  s.MarkerEdgeColor = shade(1)*cS;
  s = scatter(r,motif.sp(:,2),0.1,'.');
  s.MarkerEdgeColor = shade(2)*cS;
  s = scatter(r,motif.sp(:,3),0.1,'.');
  s.MarkerEdgeColor = shade(3)*cS;

  s = plot(r,motif.sg,'linewidth',0.1);
  s.Color = 'k';

  ylim([0,1]);
  xlim(xl);
  ylabel('Synergy')
  xticklabels({});
  ax.Color = 'none';

  ys = 1 - ymargin/2 - 3.5*h - 2*ysp;

  %% Motif Strength (Additive)
  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  shade = linspace(0.4,0.8,3);
  s = scatter(r,motif.ap(:,1),0.1,'.'); hold on;
  s.MarkerEdgeColor = shade(1)*cA;
  s = scatter(r,motif.ap(:,2),0.1,'.');
  s.MarkerEdgeColor = shade(2)*cA;
  s = scatter(r,motif.ap(:,3),0.1,'.');
  s.MarkerEdgeColor = shade(3)*cA;

  s = plot(r,motif.ag,'linewidth',0.1);
  s.Color = 'k';

  ylim([0,1]);
  xlim(xl);
  ylabel('Anti-Synergy')
  xticklabels({});
  ax.Color = 'none';

  % Weight Transformation

  xhm = (w + xmargin - (nhm+1)*hmsp) / nhm;
  ys = 1 - ymargin/2 - 1.5*h;

  xs = xs - xmargin/2 + hmsp;
  for i=1:nhm
    pos = [xs,ys + h+ysp,xhm,xhm];
    ax = axes('pos',pos);
    indx = floor((i-1)*numel(W)/nhm);
    p = imagesc(W{indx+1});
    p.AlphaData = 0.8;
    colormap(ax, cmap2);
    caxis([0,1]);
    axis square;
    axis off;
    xs = xs + xhm + hmsp;
  end

  pos_leg = pos;
  pos_leg = pos_leg + [xhm/1.5 -xhm*0.85 0 0];
  pos_leg(3) = xhm/4;
  pos_leg(4) = xhm/1.5;
  ax = axes('pos',pos_leg);
  p = imagesc([0 1], [0 1], flipud(linspace(0,1,1024)'));
  p.AlphaData = 0.8;
  ylim([0,1]);
  ax.XAxis.Visible = 'off';
  yticklabels({'1', '', '0'});
  colormap(ax,cmap2);
  caxis([0,1]);

  % Muscle strength
  xs = xmargin/2;
  ys = 1 - ymargin/2 - 3.5*h - 3*ysp - 0.5*h;
  pos = [xs,ys,w,0.5*h];
  ax = axes('position',pos);
  s = scatter(r,muscle_strength,1,'.');
  s.MarkerEdgeColor = 'k';
  ylim([0,1]);
  ylabel('Strength')
  xlabel('sim steps')
  ax.Color = 'none';
  xlim(xl);


  xs = 0.01;
  ys = 0.02;
  ann = annotation('textbox','Position',[xs,ys,0.5,0.2]);
  ann.Interpreter = 'none';
  ann.String = keys_str;
  ann.EdgeColor = 'none';
  ann.HorizontalAlignment = 'left';
  ann.VerticalAlignment = 'bottom';
  ann.FontSize = 8;

  ann = annotation('textbox','Position',[xs,ys,0.4,0.2]);
  ann.Interpreter = 'none';
  ann.String = vals_str;
  ann.EdgeColor = 'none';
  ann.HorizontalAlignment = 'right';
  ann.VerticalAlignment = 'bottom';
  ann.FontSize = 8;


  ann = annotation('textbox','Position',[xs,ys,0.4,0.15]);
  ann.Interpreter = 'none';
  ann.String = name;
  ann.EdgeColor = 'none';
  ann.HorizontalAlignment = 'center';
  ann.VerticalAlignment = 'top';
  ann.FontSize = 8;


  xs = 0.50;
  ys = 0.0;
  h = 0.15;
  w = 0.15;
  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  Rm = reshape(sum(motif.mask(:,1:3),2),6,6);
  Sm = reshape(sum(motif.mask(:,4:9),2),6,6);
  Am = reshape(sum(motif.mask(:,10:end),2),6,6);

  p = imagesc(cvR*Rm + cvS*Sm + cvA*Am);
  p.AlphaData = 0.8;
  axis square;
  caxis([-1,1]);
  colormap(ax,parula);
  ax.XAxisLocation = 'top';
  yticks(1:6);
  xticks(1:6);
  yticklabels({'MN-SF', 'MN-SE', 'MN-EF', 'MN-EE', 'MN-BF', 'MN-BE'});
  xticklabels({'Ia-SF', 'Ia-SE', 'Ia-EF', 'Ia-EE', 'Ia-BF', 'Ia-BE'});
  ax.XTickLabelRotation = 45;
  ax.XAxis.FontSize = 7;
  ax.YAxis.FontSize = 7;

  xs = 0.01;
  ys = 0.125;
  h = 0.1;
  w = 0.1;

  xbasissp = 0.060;
  ys = ys + h*0.6;
  w = w/2.5;
  h = w;
  lab = {'S','E','B'};
  shade = linspace(0.4,0.8,3);

  pos = [xs,ys,w,h];
  ax = axes('position',pos);

  Rm = reshape(sum(motif.mask(:,1:3),2),6,6);
  Sm = reshape(sum(motif.mask(:,4:9),2),6,6);
  Am = reshape(sum(motif.mask(:,10:end),2),6,6);

  p = imagesc(cvR*Rm + cvS*Sm + cvA*Am);
  p.AlphaData = 0.8;
  axis square;
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

  xs = xs + w + xbasissp;
  for i=1:3
    pos = [xs,ys,w,h];
    ax = axes('position',pos);

    Rmi = reshape(motif.maskp(:,i),6,6);

    dark_cR = shade(i) * cR;  % darken by scaling RGB toward black
    rgb = zeros(6, 6, 3);
    for ch = 1:3
        rgb(:,:,ch) = dark_cR(ch) * double(Rmi);
    end
    rgb(repmat(~Rmi, 1, 1, 3)) = NaN;

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

    ann = annotation('textbox',pos+[-0.065,0,0.1,0]);
    if i==1
      ann.String = [' = r_',lab{i}];
    else
      ann.String = [' + r_',lab{i}];
    end
    ann.EdgeColor ='none';
    ann.HorizontalAlignment = 'left';
    ann.VerticalAlignment = 'middle';
    ann.FontSize = 7;

    xs = xs + w + xbasissp;
  end

  for i=1:3
    pos = [xs,ys,w,h];
    ax = axes('position',pos);

    Smi = reshape(motif.maskp(:,i+3),6,6);

    dark_cS = shade(i) * cS;  % darken by scaling RGB toward black
    rgb = zeros(6, 6, 3);
    for ch = 1:3
        rgb(:,:,ch) = dark_cS(ch) * double(Smi);
    end
    rgb(repmat(~Smi, 1, 1, 3)) = NaN;
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

    ann = annotation('textbox',pos+[-0.065,0,0.1,0]);
    ann.String = [' + s_',lab{i}];
    ann.EdgeColor ='none';
    ann.HorizontalAlignment = 'left';
    ann.VerticalAlignment = 'middle';
    ann.FontSize = 7;

    xs = xs + w + xbasissp;
  end
  for i=1:3
    pos = [xs,ys,w,h];
    ax = axes('position',pos);

    Ami = reshape(motif.maskp(:,i+6),6,6);

    dark_cA = shade(i) * cA;  % darken by scaling RGB toward black
    rgb = zeros(6, 6, 3);
    for ch = 1:3
        rgb(:,:,ch) = dark_cA(ch) * double(Ami);
    end
    rgb(repmat(~Ami, 1, 1, 3)) = NaN;
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

    ann = annotation('textbox',pos+[-0.065,0,0.1,0]);
    ann.String = [' + a_',lab{i}];
    ann.EdgeColor ='none';
    ann.HorizontalAlignment = 'left';
    ann.VerticalAlignment = 'middle';
    ann.FontSize = 7;

    xs = xs + w + xbasissp;
  end


  D = [mean(motif.rp(end-200:end,:),1), ...
        mean(motif.sp(end-200:end,:),1), ...
        mean(motif.ap(end-200:end,:),1)]';

  C = [shade(1).*cR; shade(2).*cR; shade(3).*cR; ...
        shade(1).*cS; shade(2).*cS; shade(3).*cS; ...
        shade(1).*cA; shade(2).*cA; shade(3).*cA];

  xs = 0.70;
  ys = 0.025;
  w = 0.20;
  h = 0.10;
  pos = [xs, ys, w, h];
  ax = axes('Position',pos);
  b = barh(D, 'LineWidth', 0.5, 'FaceColor', 'flat');
  b.CData = C;
  b.EdgeColor = 'none';
  grid on;
    xlim([0,1]);
    xticks(linspace(0,1,5));
    xticklabels({0, '', '', '', 1});
    yticklabels({'r_S', 'r_E', 'r_B', 's_S', 's_E', 's_B', 'a_S', 'a_E', 'a_B'});
    ax.YAxis.FontSize = 7;
    ax.XAxis.FontSize = 7;
  ax.Color = 'none';

end
