function make_weight_plots(S,W,i,j,prominance)

fs = 200;
dt = 1/fs;

w =@(i,j) arrayfun(@(t) W{t}(i,j),1:numel(W));

wij = w(i,j);
dw = gradient(wij)/dt;
dw_pos = (dw + abs(dw)) / 2;
[pks,loc] = findpeaks(dw_pos,'MinPeakProminence',prominance);

figure;
ax1 = subplot(211);
plot(wij,'k'); hold all;
scatter(loc,wij(loc),10,'r');
ax2 = subplot(212);
plot(dw,'k'); hold all;
scatter(loc,pks,10,'r');
linkaxes([ax1,ax2],'x');

winSec = 1;
halfwin = 1/dt;
win = [-halfwin:halfwin];

quiry = {'qpos','qvel','len','vel','drive','act_Ia','act_MN'};
muscle_lab = {'SF','SE','EF','EE','BF','BE'};

nq = numel(quiry);

figure('Position',[1440 818 1514 310]);

sp = 0.001;
xsp = 0.04;
ysp = 0.05;

w = (1 - (1+nq)*xsp)/nq;
xs = xsp/2;

for q=1:nq;
  nsig = size(S.(quiry{q}),2);
  h = (1 - 2*ysp - (nsig-1)*sp) / nsig;

  ys = 1 - ysp - h;
  axs = gobjects(nsig+1,1);

  for n=1:nsig
    axs(n) = axes('Position',[xs ys w h]);

    x = S.(quiry{q})(:,n);
    xq = x(win + loc');
    xq_mu = mean(xq);

    plot(win,xq,'Color',[0 0 0 0.1]); hold all;
    plot(win,xq_mu,'r');
    yl = ylim;
    plot([0 0],yl,'--k');

    axs(n).XAxis.Visible = 'off';
    axs(n).YAxis.Visible = 'off';
    axs(n).Color = 'none';
    grid on;

    if (q==6) && (n==j)
      ar = annotation('arrow','Position',[xs-3*xsp/4,ys+h/3.5,0.01,0]);
      ar.LineStyle = 'none';
    end

    if (q==7) && (n==i)
      ar = annotation('arrow','Position',[xs-3*xsp/4,ys+h/3.5,0.01,0]);
      ar.LineStyle = 'none';
    end

    if nsig==6
      ann = annotation('textbox','Position',[xs-xsp/2,ys-h/2,w,h]);
      ann.String = muscle_lab{n}; 
      ann.Interpreter = 'none';
      ann.EdgeColor = 'none';
      ann.FontSize = 10;
    end

    if nsig==2
      if n==1
      ann = annotation('textbox','Position',[xs-xsp/2,ys-h/2,w,h]);
      ann.String = 'S';
      ann.Interpreter = 'none';
      ann.EdgeColor = 'none';
      ann.FontSize = 10;
      end
      if n==2
        ann = annotation('textbox','Position',[xs-xsp/2,ys-h/2,w,h]);
        ann.String = 'E';
        ann.Interpreter = 'none';
        ann.EdgeColor = 'none';
        ann.FontSize = 10;

        ann = annotation('textbox','Position',[xs+0.7*w,ys,0.3,h]);
        ann.String = '\uparrow Flexion';
        ann.EdgeColor = 'none';
        ann.VerticalAlignment = 'top';
        ann.FontSize = 10;

        ann = annotation('textbox','Position',[xs+0.7*w,ys,0.3,h]);
        ann.String = '\downarrow Extension';
        ann.EdgeColor = 'none';
        ann.VerticalAlignment = 'bottom';
        ann.FontSize = 10;
      end
    end
 
    if n==1
      ann = annotation('textbox','Position',[xs,ys+0.05,w,h]);
      %ann.String = {label,quiry{q}};
      ann.String = quiry{q};
      ann.Interpreter = 'none';
      ann.EdgeColor = 'none';
      ann.FontSize = 14;
    end

    if n==nsig
      axs(n+1) = axes('Position',[xs ys w 0.1]);
      plot(0);
      xlim([-halfwin,halfwin]);
      axs(n+1).Color = 'none';
      axs(n+1).YAxis.Visible = 'off';
      box off;
    end

    ys = ys - sp - h;
  end
  xs = xs + xsp + w;
end

linkaxes(axs,'x');



end
