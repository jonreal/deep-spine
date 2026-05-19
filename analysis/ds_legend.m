  % Top Row: Legend, Motif and Rank 1 summary
  %
  ncol = 5;
  h = 0.1;
  w = (1 - xmargin - (ncol-1)*xsp)/ncol;


  % Motif Legend
  xs = xmargin/2;
  ys = 1 - ymargin/2 - h;

  pos = [xs,ys,w,h];
  ax = axes('position',pos);
  p = imagesc( ...
    +cvR*double(motif.mask.R) ...
    + cvS*double(motif.mask.S) ...
    + cvA*double(motif.mask.A));
  p.AlphaData = 0.8;
  axis square;
  axis off;
  caxis([-1,1]);

  legend_list = { {cvR, 'Reflex'}, ...
                  {cvS, 'Synergy'}, ...
                  {cvA, 'Anti-synergy'} };

  xs = xs + w/2 + 4*xsp;
  ys = ys + h - h/3.5;
  wl = 0.03;
  hl = 0.03;
  for i = 1:length(legend_list)
    pos = [xs,ys,wl,hl];
    ax = axes('Position',pos);
    p = imagesc(legend_list{i}{1});
    p.AlphaData = 0.8;
    axis off;
    axis square;
    yticklabels([])
    xticklabels([])
    caxis([-1,1]);

    ann = annotation('textbox','Position',[xs+2*xsp ys 0.1 h/3]);
    ann.String = legend_list{i}{2};
    ann.EdgeColor = 'none';
    ann.HorizontalAlignment = 'left';
    ann.VerticalAlignment = 'middle';

    ys = ys - h/3;

  end


  % SVD Legend
  xs = xmargin/2 + 2*w + 2*xsp;
  ys = 1 - ymargin/2 - h;
  wsv = w/2;
  hsv = h/2;

  ex_synergy = ds_svd({0.1*rand(6,6) + 0.5*diag(rand(6,1))});

  for i=1:6
    pos = [xs,ys,wsv,hsv];
    ax = axes('position',pos);
    ex_synergy.rank1_mats{1}{1}
    p = imagesc(ex_synergy.rank1_mats{1}{i})
    p.AlphaData = 0.8;
    axis square;
    axis off;
    caxis([-1,1]);
    xs = xs + wsv;
  end
