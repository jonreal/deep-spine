function make_all_weights_timeseries(W)

wfun =@(i,j) arrayfun(@(t) W{t}(i,j),1:numel(W));

c1 = ([242 201 98] ./ 256) * 0.75;
c2 = ([114 198 173] ./ 256) * 0.75;
c3 = ([89 72 172] ./ 256) * 0.75;

figure;
xsp = 0.01;
xlab = 0.1;
ysp = 0.01;
ylab = 0.1;

w = (1 - xlab - 5*xsp) / 6;
h = (1 - ylab - 5*ysp) / 6;
xs = xlab/2;
ys = 1 - ylab/2 - h;

for i=1:6
  for j=1:6
    wij = wfun(i,j);
    pos = [xs,ys,w,h];
    ax = axes('Position',pos);
    p = plot(wij,'k');
    ylim([0,1]);
    if i==j
      p.Color = c1;
    elseif mod(i+j,2) == 0
      p.Color = c2;
    else
      p.Color = c3;
    end
    box on;
    xticklabels([]);
    yticklabels([]);
    ax.Color = 'none';
    xs = xs + xsp + w;
  end
  xs = xlab/2;
  ys = ys - h - ysp;
end
end
