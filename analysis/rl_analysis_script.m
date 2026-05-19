


% BEST RUN
%paths = {
%  '../../experiments/hebbian/logs/pendulum-compare-11/df.mat'}

paths = {
  '../../experiments/hebbian/logs/PNAS/rl-baseline/df.mat'}

fs = 200;
decimate = 10;

% step| eps
abscissa = 'Episode';



T = table();
for i=1:numel(paths)
  S = load(paths{i});

  S.table.Step = S.table.Step(:);
  S.table.Value = S.table.Value(:);
  S.table.Tag = S.table.Tag(:);
  S.table.ModelID = S.table.ModelID(:);
  S.table.Model = S.table.Model(:);

  Ti = struct2table(S.table);
  T = [T; Ti];

end

all_models = unique(T.ModelID);


figure('Position',[1440 818 556 400]);

w1 = 0.50;
w2 = 0.25;

xsp = 0.075;
ysp = 0.1;
w = w1;
h = 0.22
xs = 0.1;
ys = 0.95 - h;

pos = [xs,ys,w,h];
ax = axes(Position=pos);

bl = [];
bl2 = [];
sp = [];
sp2 = [];
for i=1:numel(all_models)
  % episode #
  iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/solved');
  [step, idx] = sort(T.Step(iq));      % or T.WallTime / T.Time
  solved = T.Value(iq);
  solved = solved(idx);

  iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/eps');
  [step, idx] = sort(T.Step(iq));      % or T.WallTime / T.Time
  eps_ = T.Value(iq);
  eps_ = eps_(idx);

  if strcmp(abscissa,'Episode')
    x = eps_;
  else
    x = step;
  end

  p = plot(x, solved);
  hold on;

  maxval = max(solved);

  model_type = unique(T.Model(iq));
  if strcmp(model_type,'identity')
    p.Color = [0 0 0 0.5];
    bl(end+1) = maxval;
  else
    p.Color = [1 0 0 0.5];
    sp(end+1) = maxval;
  end
end
ylabel('# Perfect Episodes');
xlabel(abscissa);

xs = xs + xsp + w;
w = w2;

pos = [xs,ys,w,h];
ax = axes(Position=pos);


  p = scatter(1.25 + 0.2*[1:numel(bl(:))]/numel(bl), bl(:));
  p.MarkerFaceColor = [0 0 0];
  p.MarkerFaceAlpha = 0.5;
  p.MarkerEdgeColor = 'none';

  hold on;
  p = scatter(1.75 - 0.2*[1:numel(sp(:))]/numel(sp), sp(:));
  p.MarkerFaceColor = [1 0 0];
  p.MarkerFaceAlpha = 0.5;
  p.MarkerEdgeColor = 'none';

  hold on;

  bp = boxplot([bl(:),sp(:)], ...
    'Widths',0.1, ...
    'Colors',['k','r']);
  xticklabels({'end-to-end','spine'})
ylabel({'# of Perfect', 'Episodes'});


xs = 0.1
w = w1
ys = ys - h - ysp;

pos = [xs,ys,w,h];
ax = axes(Position=pos);


bl = [];
bl2 = [];
sp = [];
sp2 = [];
for i=1:numel(all_models)

  % episode #
  iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/ep_rew_mean_10');
  [step, idx] = sort(T.Step(iq));      % or T.WallTime / T.Time
  val = T.Value(iq);
  val = val(idx);

  iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/eps');
  [step, idx] = sort(T.Step(iq));      % or T.WallTime / T.Time
  eps_ = T.Value(iq);
  eps_ = eps_(idx);

  if strcmp(abscissa,'Episode')
    x = eps_;
  else
    x = step;
  end

  p = plot(x, val);

  hold on;

  [mval,mv] = max(val);

  model_type = unique(T.Model(iq));
  if strcmp(model_type,'identity')
    p.Color = [0 0 0 0.5 + 0.4*(i-1)/numel(all_models)];
    %pc.MarkerFaceColor = 'none';
    %pc.MarkerEdgeColor = [0 0 0];
    %pc.MarkerSize = 6;
    bl(end+1) = mval;
    bl2(end+1) = mv;
  else
    p.Color = [1 0 0 (i-1)/numel(all_models)];
    %pc.MarkerFaceColor = 'none';
    %pc.MarkerEdgeColor = [1 0 0];
    %pc.MarkerSize = 6;
    sp(end+1) = mval;
    sp2(end+1) = mv;
  end
end
xlabel(abscissa);
ylabel('Mean Reward');
%ax.YAxis.Exponent = 3;
%

xs = xs + xsp + w;
w = w2;
pos = [xs,ys,w,h];
ax = axes(Position=pos);

  p = scatter(1.25 + 0.2*[1:numel(bl(:))]/numel(bl), bl(:));
  p.MarkerFaceColor = [0 0 0];
  p.MarkerFaceAlpha = 0.5;
  p.MarkerEdgeColor = 'none';

  hold on;
  p = scatter(1.75 - 0.2*[1:numel(sp(:))]/numel(sp), sp(:));
  p.MarkerFaceColor = [1 0 0];
  p.MarkerFaceAlpha = 0.5;
  p.MarkerEdgeColor = 'none';

  hold on;

  bp = boxplot([bl(:),sp(:)], ...
    'Widths',0.1, ...
    'Colors',['k','r']);
  xticklabels({'end-to-end','spine'})
ylabel({'Mean', 'Reward'});

%ax.YAxis.Exponent = 3;

%return
%
%figure;
%  p = scatter(1.25 + 0.01*[1:numel(bl2(:))], bl2(:));
%  p.MarkerFaceColor = [0 0 0];
%  p.MarkerFaceAlpha = 0.5;
%  p.MarkerEdgeColor = 'none';
%
%  hold on;
%  p = scatter(1.75 + 0.01*[1:numel(sp2(:))], sp2(:));
%  p.MarkerFaceColor = [1 0 0];
%  p.MarkerFaceAlpha = 0.5;
%  p.MarkerEdgeColor = 'none';
%
%  hold on;
%
%  bp = boxplot([bl2(:),sp2(:)], ...
%    'Widths',0.1, ...
%    'Colors',['k','r']);
%  xticklabels({'end-to-end','spine'})
%ylabel('First Perfect');


xs = 0.1
w = w1;
ys = ys - h - ysp;

pos = [xs,ys,w,h];
ax = axes(Position=pos);

bl = [];
sp = [];
for i=1:numel(all_models)
  iq = strcmp(T.ModelID, all_models{i}) & strcmp(T.Tag,'eps/eps');
  [step, idx] = sort(T.Step(iq));      % or T.WallTime / T.Time
  step = double(step);
  val = T.Value(iq);
  val = val(idx);
  eps_ = val;
  survive = [0; diff(step)*(decimate/fs)];

  if strcmp(abscissa,'Episode')
    x = eps_;
  else
    x = step;
  end

  p = plot(x,survive);
  hold on;
  model_type = unique(T.Model(iq));
  if strcmp(model_type,'identity')
    p.Color = [0 0 0 0.1];
    %bl(end+1) = mean(maxk(val,20));
    bl(end+1) = mean(survive);
    %bl(end+1) = mean(val);
  else
    p.Color = [1 0 0 0.1];
    sp(end+1) = mean(survive);
    %sp(end+1) = mean(val);
  end

  p = plot(x,movmean(survive,100));
  model_type = unique(T.Model(iq));
  if strcmp(model_type,'identity')
    p.Color = [0 0 0 0.5];
  else
    p.Color = [1 0.1 0.1 0.5];
  end
end
ylabel('Survival Time');
xlabel(abscissa);

xs = xs + xsp + w;
w = w2;
pos = [xs,ys,w,h];
ax = axes(Position=pos);

%for i=1:numel(all_models)
%  iq = strcmp(T.ModelID,all_models(i)) & strcmp(T.Tag,'eps/eps');
%  step = sort(T.Step(iq));
%  val = gradient(double(sort(T.Step(iq))));
%  p = plot(step,val);
%  model_type = unique(T.Model(iq));
%  if strcmp(model_type,'identity')
%    p.Color = [0 0 0 0.05 + 0.15*(i/numel(all_models))];
%  else
%    p.Color = [1 0 0 0.05 + 0.15*(i/numel(all_models))];
%  end
%
%  p = plot(step,movmean(val,5));
%  model_type = unique(T.Model(iq));
%  if strcmp(model_type,'identity')
%    p.Color = [0 0 0 0.2 + 0.8*(i/numel(all_models))];
%  else
%    p.Color = [1 0.1 0.1 0.2 + 0.8*(i/numel(all_models))];
%  end
%end
%ylabel('Survival Time');
%xlabel('Training Step');


  p = scatter(1.25 + 0.2*[1:numel(bl(:))]/numel(bl), bl(:));
  p.MarkerFaceColor = [0 0 0];
  p.MarkerFaceAlpha = 0.5;
  p.MarkerEdgeColor = 'none';

  hold on;
  p = scatter(1.75 - 0.2*[1:numel(sp(:))]/numel(sp), sp(:));
  p.MarkerFaceColor = [1 0 0];
  p.MarkerFaceAlpha = 0.5;
  p.MarkerEdgeColor = 'none';

  hold on;

  bp = boxplot([bl(:),sp(:)], ...
    'Widths',0.1, ...
    'Colors',['k','r']);
  xticklabels({'end-to-end','spine'})
  ylabel({'Mean','Survival Time'});


return



figure; hold all;
for i=1:numel(all_models)
  % grad(episode #)
  iq = strcmp(T.ModelID,all_models(i)) & strcmp(T.Tag,'eps/eps');
  steps = sort(T.Step(iq));
  val = sort(T.Value(iq));
  p = plot(steps, movmean(gradient(val)./gradient(double(steps)),100));
  p.Color = [0 0 0 (i-1)/numel(all_models)];
  xlabel('Training Steps');
  ylabel('Number of Episodes');
end
return


figure; hold all;
for i=1:numel(all_models)
  iq = strcmp(T.ModelID,all_models(i)) & strcmp(T.Tag,'eps/eps');
  step = sort(T.Step(iq));
  val = gradient(double(sort(T.Step(iq))));
  num = numel(val);
  p = plot(val);
  p.Color = [0 0 0 (i-1)/numel(all_models)];
end

figure; hold all;
for i=1:numel(all_models)
  iq = strcmp(T.ModelID,all_models(i)) & strcmp(T.Tag,'eps/eps');
  step = sort(T.Step(iq));
  val = gradient(double(sort(T.Step(iq))));
  num = numel(val);
  p = plot(movmean(val,2));
  p.Color = [0 0 0 (i-1)/numel(all_models)];
  i0 = floor(num - 0.5*num);
end


figure; hold all;
for i=1:numel(all_models)
  iq = strcmp(T.ModelID,all_models(i)) & strcmp(T.Tag,'eps/eps');
  step = sort(T.Step(iq));
  val = sort(T.Step(iq));
  num = numel(val);
  p = plot(val);
  p.Color = [0 0 0 (i-1)/numel(all_models)];
end


