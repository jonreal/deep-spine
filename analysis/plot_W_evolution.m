function plot_W_evolution(W,motif,opt)
  arguments
    W
    motif
    opt.w = 0.8
    opt.nhm = 10
    opt.hmsp = 0.1
    opt.ys = 0.1;
    opt.xs = 0.1;
    opt.h = 0.01;
  end

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



  xhm = (opt.w+4*opt.hmsp - (opt.nhm-1)*opt.hmsp) / opt.nhm;
  xs = opt.xs - 2*opt.hmsp;
  ys = opt.ys + opt.h - xhm/2;
  for i=1:opt.nhm
    pos = [xs,ys,xhm,xhm];
    ax = axes('pos',pos);
    indx = floor((i-1)*numel(W)/opt.nhm);
    p = imagesc(W{indx+1});
    p.AlphaData = 0.8;
    colormap(ax, cmap2);
    caxis([0,1]);
    axis square;
    axis off;
    xs = xs + xhm + opt.hmsp;
  end

  pos_leg = pos;
  pos_leg = pos_leg + [2*xhm 0 0 0];
  pos_leg(3) = xhm/4;
  pos_leg(4) = xhm;
  ax = axes('pos',pos_leg);
  p = imagesc([0 1], [0 1], flipud(linspace(0,1,1024)'));
  p.AlphaData = 0.8;
  ylim([0,1]);
  ax.XAxis.Visible = 'off';
  yticklabels({'1', '', '0'});
  colormap(ax,cmap2);
  caxis([0,1]);

  %% Motif Strength (Additive)
  xs = opt.xs;
  ys = opt.ys;
  h = opt.h - xhm;
  pos = [xs,ys,opt.w,h];
  ax = axes('position',pos);

  n = numel(motif.rg);
  r = 1:n;

  shade = linspace(0.4,0.8,3);
  s = scatter(r,motif.rg,0.01,'.'); hold on;
  s.MarkerEdgeColor = cR;
  s = scatter(r,motif.sg,0.01,'.');
  s.MarkerEdgeColor = cS;
  s = scatter(r,motif.ag,0.01,'.');
  s.MarkerEdgeColor = cA;

  ylim([0,1]);
  xlim([1 n]);
  ylabel('Motif Strength')
  xticklabels({});
  ax.Color = 'none';


end
