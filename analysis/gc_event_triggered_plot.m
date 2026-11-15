function gc_event_triggered_plot(S,W,i,j,opt)
  arguments
    S
    W
    i
    j
    opt.pct (1,1) double = 100
    opt.loFrac (1,1) double = 0.2
    opt.hiFrac (1,1) double = 0.9
    opt.alpha = 0.025;
    opt.winSec = 1;
    opt.lag = 0.5;
  end

fs = 200;
dt = 1/fs;

[loc, pks, wij, dw] = gc_weight_events(W, i, j, pct=opt.pct, fs=fs, ...
                                       loFrac=opt.loFrac, hiFrac=opt.hiFrac);


figure;

ax1 = subplot(211);
plot(wij,'k'); hold all;
scatter(loc,wij(loc),10,'r');
ax2 = subplot(212);
plot(dw,'k'); hold all;
scatter(loc,pks,10,'r');
linkaxes([ax1,ax2],'x');

halfwin = (opt.winSec/2)/dt;
win = [-halfwin:halfwin];

quiry = {'qpos','qvel','len','vel','drive','act_MN','act_Ia'};
labels = {'Joint Position','Joint Velocity', 'Length', 'Velocity', 'Babble', ...
  'Motoneuron', 'Ia Neuron'};

ylims = {[-0.1,0.1],[-1,1],[0.45,0.55],[-0.1,0.1],[0,1],[0,1],[0,1]};


quiry = {'qpos','qvel','drive','act_MN','act_Ia'};
labels = {'Position','Velocity','Babble', ...
  'Motoneuron', 'Ia Neuron'};
ylims = {[-5,5],[-50,50],[0,1],[0,1],[0,1]};


muscle_lab = {'SF','SE','EF','EE','BF','BE'};


nq = numel(quiry);

figure('Position',[1440 818 767 237]);

xsp = 0.05;
ysp = 0.025;
xmargin = 0.15;
ymargin = 0.25;

w = (1 - xmargin - (nq-1)*xsp)/nq;
xs = xmargin/2;

for q=1:nq;
  nsig = size(S.(quiry{q}),2);
  h = (1 - ymargin - (nsig-1)*ysp) / nsig;

  ys = 1 - ymargin/2 - h;

  for n=1:nsig
    ax = axes('Position',[xs ys w h]);

    if q == 1 || q == 2
      x = rad2deg(S.(quiry{q})(:,n));
    else
      x = S.(quiry{q})(:,n);
    end
    xq = x(win + loc');
    xq_mu = mean(xq);

    plot((win*dt)*1000,xq,'Color',[0 0 0 opt.alpha]); hold all;
    plot((win*dt)*1000,xq_mu,'r');
    ylim(ylims{q})
    plot([0 0],ylims{q},'--k','LineWidth',0.5);

    ax.Color = 'none';
    grid off;
    box off;
    ax.TickDir = 'out';
    yl_ = ylims{q};
    yticks(linspace(yl_(1),yl_(2),3));
    yticklabels({yl_(1),'',yl_(2)});

    if n ~= nsig
      %ax.YAxis.Visible = 'off';
      yticklabels({});
      ax.XAxis.Visible = 'off';
      %xticklabels({});
    else
      if nq == 7
        if q == 1
          ylabel('(deg)')
        end
        if q == 2
          ylab = ylabel('(deg/s)');
        end
        if q == 4
          xlabel('Time from weight-increase event (ms)')
        end
        if q == 5 || q == 6 || q == 7
          ylabel('(a.u.)')
        end
      end
      if nq == 5
        if q == 1
          ylab = ylabel('(rad)');
          ylab.Units = 'normalized';
          ylab.Position(1) = ylab.Position(1) + 0.1;
          xticklabels({''})
        end
        if q == 2
          ylab = ylabel('(rad/s)');
          ylab.Units = 'normalized';
          ylab.Position(1) = ylab.Position(1) + 0.12;
          xticklabels({''})
        end
        if q == 3 || q == 4 || q == 5
          if q == 3
            xlabel('Time from weight-increase event (ms)')
          else
            xticklabels({''})
          end
        end
      end
    end


    %yl = ylim;
    %ax = axes('Position',[xs-0.001,ys,0.01,h]);
    %plot(0);
    %ylim(yl);
    %yticks(linspace(yl(1),yl(2),3));
    %yticklabels({yl(1),'',yl(2)});
    %ax.XAxis.Visible = 'off';
    %ax.Color = 'none';
    %box off;

    if strcmp(quiry(q),'act_Ia') && (n==j)
      ar = annotation('arrow','Position',[xs-0.02,ys+h/2,0.01,0]);
      ar.LineStyle = 'none';
    end

    if strcmp(quiry(q),'act_MN') && (n==i)
      ar = annotation('arrow','Position',[xs-0.02,ys+h/2,0.01,0]);
      ar.LineStyle = 'none';
    end

    if (nsig==6 && q == 7 && nq == 7) || (nsig==6 && q == 5 && nq == 5) 
      ann = annotation('textbox','Position',[xs+0.02,ys,w,h]);
      ann.String = muscle_lab{n};
      ann.Interpreter = 'none';
      ann.EdgeColor = 'none';
      ann.FontSize = 10;
      ann.HorizontalAlignment = 'Right';
      ann.VerticalAlignment = 'Middle';
    end

    if nsig==2
      if n==1
        ann = annotation('textbox','Position',[xs,ys,w,h]);
        ann.String = 'S';
        ann.Interpreter = 'none';
        ann.EdgeColor = 'none';
        ann.FontSize = 10;
        ann.HorizontalAlignment = 'Left';
        ann.VerticalAlignment = 'Top';
      end
      if n==2
        ann = annotation('textbox','Position',[xs,ys,w,h]);
        ann.String = 'E';
        ann.Interpreter = 'none';
        ann.EdgeColor = 'none';
        ann.FontSize = 10;

        ann = annotation('textbox','Position',[xs+0.5*w,ys,0.3,h]);
        ann.String = '\uparrow Flx.';
        ann.EdgeColor = 'none';
        ann.VerticalAlignment = 'top';
        ann.FontSize = 10;

        ann = annotation('textbox','Position',[xs+0.5*w,ys,0.3,h]);
        ann.String = '\downarrow Ext.';
        ann.EdgeColor = 'none';
        ann.VerticalAlignment = 'bottom';
        ann.FontSize = 10;
      end
    end
 
    if n==1
      ann = annotation('textbox','Position',[xs,ys+0.07,w,h]);
      %ann.String = {label,quiry{q}};
      ann.String = labels{q};
      ann.Interpreter = 'none';
      ann.EdgeColor = 'none';
      ann.FontSize = 10;
    end

    %if n==nsig
    %  axs(n+1) = axes('Position',[xs ys w 0.1]);
    %  plot(0);
    %  xlim([-halfwin,halfwin]);
    %  axs(n+1).Color = 'none';
    %  axs(n+1).YAxis.Visible = 'off';
    %  box off;
    %end

    ys = ys - ysp - h;
  end
  xs = xs + xsp + w;
end


end
