% first load ./logs/exp9/trials/1756957634/trial_data.mat

fs = 200;
dt = 1/fs;

w =@(i,j) arrayfun(@(t) W{t}(i,j),1:numel(W));

doCase = 3;

if doCase == 1
% Ia_BE -> MN_BE
  w66 = w(6,6);
  dw66 = gradient(w66)/dt;
  tq = [40000, 170000];
  [pks,loc] = findpeaks(dw66, 'MinPeakProminence',5e-4);
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
tq = [230000, 310000];
[pks,loc] = findpeaks(dw24, 'MinPeakProminence',5e-4);
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
% Ia_BE -> MN_EE
w46 = w(4,6);
dw46 = gradient(w46)/dt;
tq = [415000, 490000];
[pks,loc] = findpeaks(dw46, 'MinPeakProminence',5e-4);
mask = loc > tq(1) & loc <= tq(end);
loc = loc(mask);
pks = pks(mask);
figure;
  subplot(211);
  plot(w46,'k'); hold all;
  plot([tq(1),tq(1)],ylim,'--k');
  plot([tq(end),tq(end)],ylim,'--k');
  ylabel('weight')
  subplot(212);
  plot(dw46,'k');
  hold all;
  scatter(loc,pks,10,'r');
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
  [pks,loc] = findpeaks(dw24, 'MinPeakProminence',5e-4);
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
end

winSec = 1;
halfwin = 1/dt;
win = [-halfwin:halfwin];

quiry = {'qpos','qvel','len','vel','act_IIa','act_Ia','drive','act_MN','act'};
muscle_lab = {'SF','SE','EF','EE','BF','BE'};

nq = numel(quiry);

for q=1:nq;
  nsig = size(S.(quiry{q}),2);
  figure;
  sp = 0.001;
  xsp = 0.05;
  ysp = 0.05;

  w = 1 - 2*xsp;
  h = (1 - 2*ysp - (nsig-1)*sp) / nsig;

  ys = 1 - ysp - h;
  xs = xsp;
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

    if nsig==6
      ann = annotation('textbox','Position',[0,ys-h/2,w,h]);
      ann.String = muscle_lab{n}; 
      ann.Interpreter = 'none';
      ann.EdgeColor = 'none';
      ann.FontSize = 10;
    end
 
    if n==1
      ann = annotation('textbox','Position',[xs,ys+0.05,w,h]);
      ann.String = {label,quiry{q}};
      ann.Interpreter = 'none';
      ann.EdgeColor = 'none';
      ann.FontSize = 14;
    end

    if n==nsig
      axs(n+1) = axes('Position',[xs ys w 0.01]);
      plot(0);
      xlim([-halfwin,halfwin]);
      axs(n+1).Color = 'none';
      axs(n+1).YAxis.Visible = 'off';
      box off;
    end

    ys = ys - sp - h;
  end
end

linkaxes(axs,'x');

return

figure;
for i=1:numel(loc)
  plot(win,S.qpos(win + loc(i),1),'k'); hold all;
end
%yl = ylim;
%plot([], [yl(1), yl(2)],'--r');

