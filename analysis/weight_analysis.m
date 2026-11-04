
load(['../experiments/hebbian/logs/legacy/', ...
      'exp9/trials/1756957634/trial_data.mat']);

%load(['../../experiments/hebbian/logs/PNAS/homvhet/', ...
%      '1769483041/simulation_data.mat']);

return
fs = 200;
dt = 1/fs;

w =@(i,j) arrayfun(@(t) W{t}(i,j),1:numel(W));

doCase = 1;

if doCase == 1
% Ia_BE -> MN_BE
  w66 = w(6,6);
  dw66 = gradient(w66)/dt;
  tq = [1, numel(w66)];
  [pks,loc] = findpeaks(dw66);
  mask = loc > tq(1) & loc <= tq(end);
  loc = loc(mask);
  pks = pks(mask);
  figure;
  subplot(211);
  plot(w66,'k'); hold all;
  plot([tq(1),tq(1)],ylim,'--k');
  plot([tq(end),tq(end)],ylim,'--k');
  ylabel('weight')
  subplot(212);
  plot(dw66,'k'); hold all;
  scatter(loc,pks,10,'r');
  plot([tq(1),tq(1)],ylim,'--k');
  plot([tq(end),tq(end)],ylim,'--k');
  hold all;
  ylabel('d/dt')
  xlabel('step')
  label = 'Ia-BE -> MN-BE';
  sgtitle(label)

elseif doCase == 2

% Ia_EE -> MN_SE
w24 = w(2,4);
dw24 = gradient(w24)/dt;
tq = [1, numel(w24)];
[pks,loc] = findpeaks(dw24);
mask = loc > tq(1) & loc <= tq(end);
loc = loc(mask);
pks = pks(mask);
figure;
  subplot(211);
  plot(w24,'k'); hold all;
  plot([tq(1),tq(1)],ylim,'--k');
  plot([tq(end),tq(end)],ylim,'--k');
  ylabel('weight')
  subplot(212);
  plot(dw24,'k');
  hold all;
  scatter(loc,pks,10,'r');
  plot([tq(1),tq(1)],ylim,'--k');
  plot([tq(end),tq(end)],ylim,'--k');
  ylabel('d/dt')
  xlabel('step')
  label = 'Ia-EE -> MN-SE';
  sgtitle(label)


elseif doCase==3
% Ia_BE -> MN_SE
w46 = w(4,6);
dw46 = gradient(w46)/dt;
dw_pos = (dw46 + abs(dw46)) / 2;
tq = [1, numel(w46)];
[pks,loc] = findpeaks(dw_pos,'MinPeakProminence',5e-4);

mask = loc > tq(1) & loc <= tq(end);
loc = loc(mask);
pks = pks(mask);
figure;
  subplot(211);
  plot(w46,'k'); hold all;
  scatter(loc,w46(loc),10,'r','filled');
  ylabel('weight')
  subplot(212);
  plot(dw46,'k');
  hold all;
  scatter(loc,pks,10,'r','filled');
  plot([tq(1),tq(1)],ylim,'--k');
  plot([tq(end),tq(end)],ylim,'--k');
  ylabel('d/dt')
  xlabel('step')
  label = 'Ia-BE -> MN-SE';
  sgtitle(label)

elseif doCase == 4
  % Ia_SF -> MN_SF
  w24 = w(1,1);
  dw24 = gradient(w24)/dt;
  tq = [37000, 192000];
  [pks,loc] = findpeaks(dw24, 'MinPeakProminence',1e-3);
  mask = loc > tq(1) & loc <= tq(end);
  loc = loc(mask);
  pks = pks(mask);
  figure;
    subplot(211);
    plot(w24,'k'); hold all;
    plot([tq(1),tq(1)],ylim,'--k');
    plot([tq(end),tq(end)],ylim,'--k');
    ylabel('weight')
    subplot(212);
    plot(dw24,'k');
    hold all;
    scatter(loc,pks,10,'r');
    plot([tq(1),tq(1)],ylim,'--k');
    plot([tq(end),tq(end)],ylim,'--k');
    ylabel('d/dt')
    xlabel('step')
    label = 'Ia-SF -> MN-SF';
    sgtitle(label)

elseif doCase == 5
  % Ia_BF -> MN_SF
  w15 = w(1,5);
  % method 1
  %dw = gradient(w15)/dt;
  %tq = [1, numel(w15)];
  %[pks,loc] = findpeaks(dw);
  %mask = loc > tq(1) & loc <= tq(end);
  %loc = loc(mask);
  %pks = pks(mask);
  %
  %figure;
  %  ax1 = subplot(211);
  %  plot(w15,'k'); hold all;
  %  plot([tq(1),tq(1)],ylim,'--k');
  %  plot([tq(end),tq(end)],ylim,'--k');
  %  ylabel('weight')
  %  ax2 = subplot(212);
  %  plot(dw,'k');
  %  hold all;
  %  scatter(loc,pks,10,'r');
  %  plot([tq(1),tq(1)],ylim,'--k');
  %  plot([tq(end),tq(end)],ylim,'--k');
  %  ylabel('d/dt')
  %  xlabel('step')
  %  label = 'Ia-BF -> MN-SF';
  %  sgtitle(label)

  %  linkaxes([ax1,ax2],'x');
  %  return


  % method 2
  ddw = gradient(w15)/dt;
  tq = [1, numel(w15)];

  ddw_pos = (ddw + abs(ddw)) / 2;


  [pks,loc] = findpeaks(ddw_pos,'MinPeakProminence',5e-4);

  figure;
    ax1 = subplot(211);
    plot(w15,'k'); hold all;
    scatter(loc,w15(loc),10,'r');
    ax2 = subplot(212);
    plot(ddw,'k'); hold all;
    scatter(loc,pks,10,'r');

    linkaxes([ax1,ax2],'x');

end

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

return

figure;
for i=1:numel(loc)
  plot(win,S.qpos(win + loc(i),1),'k'); hold all;
end
%yl = ylim;
%plot([], [yl(1), yl(2)],'--r');

