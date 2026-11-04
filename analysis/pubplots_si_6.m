% PUBPLOTS_SI_6  SI figure: musculoskeletal plant.
%
% Script. Plots analysis/si_plant.mat, written from the MuJoCo model by
%   python analysis/si_plant_export.py      (repo root, venv active)
%
% Layout (2 x 4 tiles):
%   A  geometry at rest, arm hanging vertically: segments, joint wrapping
%      cylinders, muscle paths with origins (squares) and insertions (circles)
%      (left column, both rows)
%   B  force-length (active, passive), MuJoCo muscle model
%   C  force-velocity
%   D  normalized fiber length vs shoulder angle, operating range shaded
%   E  normalized fiber length vs elbow angle
%   F  peak isometric force per muscle for each strength ratio at lambda = 1
%      (right column, both rows)
% Moment arms are not shown: every muscle wraps a circular cylinder, so they
% are a constant 7.5 cm by design.

D = load('si_plant.mat');
names = cellstr(string(D.names));  names = strtrim(names(:))';
nm = numel(names);

fontsize = 12;
LW       = 1.5;
F_ref    = D.F0_ref;                            % 240 N
SR       = {[1 1 0.5 0.5 0.25 0.25], [1 1 0.25 0.25 0.5 0.5], [0.5 0.5 0.5 0.5 0.75 0.75]};
SRlab    = {'SR_0', 'SR_1', 'SR_2'};

% muscles: flexors warm, extensors cool, shaded by pool (S dark -> B light)
poolShade = [1.0 0.75 0.5];
cFlex = [0.80 0.33 0.10];
cExt  = [0.25 0.40 0.70];
cMus  = zeros(nm,3);
for k = 1:nm
  base = cFlex;  if mod(k,2) == 0, base = cExt; end
  s = poolShade(ceil(k/2));
  cMus(k,:) = 1 - s*(1 - base);
end
cSR = [0.2 0.2 0.2; 0.5 0.5 0.5; 0.8 0.8 0.8];

figure('Position', [50 50 1400 640], 'Color', 'w');
tl = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
panelsize = fontsize + 6;                      % panel letters
panel = @(a, lab) title(a, lab, 'Units', 'normalized', 'Position', [0.0 1.0], ...
                        'HorizontalAlignment', 'left', 'FontWeight', 'bold', ...
                        'FontSize', panelsize);

% ---------------------------------------------------
% A: geometry, arm hanging vertically (left column, both rows)
% ---------------------------------------------------
% model frame: arm along +x at rest -> rotate so it hangs along -y
rot = @(P) [P(:,2), -P(:,1)];

a = nexttile(tl, 1, [2 1]);  hold(a, 'on');  axis(a, 'equal');
J  = rot(D.joints);                             % shoulder, elbow, wrist
Cx = rot(D.cyl_xy);
base = rot([-0.30 -0.12; -0.25 -0.12; -0.25 0.12; -0.30 0.12]);
patch(a, base(:,1), base(:,2), [0.85 0.85 0.85], 'EdgeColor', 'none');   % fixed base
plot(a, J(:,1), J(:,2), '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 6);
th = linspace(0, 2*pi, 100);
for c = 1:size(Cx,1)
  plot(a, Cx(c,1) + D.cyl_r(c)*cos(th), Cx(c,2) + D.cyl_r(c)*sin(th), ...
       ':', 'Color', [0.55 0.55 0.55], 'LineWidth', 1);
end
plot(a, J(:,1), J(:,2), 'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'MarkerSize', 7);
hm = gobjects(nm,1);
for k = 1:nm
  xy = rot(densify_path(D.paths{k}, D.cyl_xy, D.cyl_r(:)));
  hm(k) = plot(a, xy(:,1), xy(:,2), '-', 'Color', cMus(k,:), 'LineWidth', LW+0.5);
  % origin (first path point) and insertion (last)
  plot(a, xy(1,1),   xy(1,2),   's', 'MarkerSize', 7, 'MarkerFaceColor', cMus(k,:), ...
       'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
  plot(a, xy(end,1), xy(end,2), 'o', 'MarkerSize', 6, 'MarkerFaceColor', cMus(k,:), ...
       'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end
ho = plot(a, NaN, NaN, 's', 'MarkerSize', 7, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');
hi = plot(a, NaN, NaN, 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');
text(a, -0.17, -0.25, '0.5 m', 'HorizontalAlignment', 'center', 'Rotation', 90, 'FontSize', fontsize-1);
text(a, -0.17, -0.75, '0.5 m', 'HorizontalAlignment', 'center', 'Rotation', 90, 'FontSize', fontsize-1);
text(a, 0.11, J(1,2), 'shoulder', 'FontSize', fontsize-1);
text(a, 0.11, J(2,2), 'elbow',    'FontSize', fontsize-1);
text(a, 0.11, J(3,2), 'wrist',    'FontSize', fontsize-1);
text(a, 0.0, -1.12, sprintf(['masses (MR_0) %.0f : %.0f kg\n' ...
     'joint range \\pm120%s'], D.body_mass(2), D.body_mass(3), char(176)), ...
     'FontSize', fontsize-1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
legend(a, [hm; ho; hi], [names, {'origin', 'insertion'}], 'Location', 'southoutside', ...
       'NumColumns', 2, 'Box', 'off', 'FontSize', fontsize-1);
xlim(a, [-0.22 0.26]);  ylim(a, [-1.25 0.33]);
a.XColor = 'none';  a.YColor = 'none';
panel(a, 'A');

% ---------------------------------------------------
% B: force-length, C: force-velocity
% ---------------------------------------------------
a = nexttile(tl, 2);  hold(a, 'on');
flMax = 2;                                      % passive reaches 1.5*fpmax = 1.95 F0 at lmax
plot(a, D.lt, D.FL_active,  'k-',  'LineWidth', LW);
plot(a, D.lt, D.FL_passive, 'k--', 'LineWidth', LW);
xlim(a, [min(D.lt) max(D.lt)]);  ylim(a, [0 flMax]);
grid(a, 'on');  box(a, 'on');  a.FontSize = fontsize;
xlabel(a, 'Fiber length (L_0)');  ylabel(a, 'Force (F_0)');
legend(a, {'active (a = 1)', 'passive'}, 'Location', 'northwest', 'Box', 'off', ...
       'FontSize', fontsize-1);
panel(a, 'B');

a = nexttile(tl, 3);  hold(a, 'on');
plot(a, D.vt, D.FV, 'k-', 'LineWidth', LW);
xline(a, 0, ':', 'Color', [0.5 0.5 0.5]);  yline(a, 1, ':', 'Color', [0.5 0.5 0.5]);
xlim(a, [-1 1]);  ylim(a, [0 1.4]);
grid(a, 'on');  box(a, 'on');  a.FontSize = fontsize;
xlabel(a, 'Velocity (L_0 v_{max}), + lengthening');  ylabel(a, 'Force (F_0)');
panel(a, 'C');

% ---------------------------------------------------
% D, E: normalized fiber length vs joint angle
% ---------------------------------------------------
jn  = {'Shoulder', 'Elbow'};
lab = {'D', 'E'};
for jnt = 1:2
  a = nexttile(tl, 5 + jnt);  hold(a, 'on');   % tiles 6, 7
  patch(a, [-120 120 120 -120], D.op_range([1 1 2 2]), [0.92 0.92 0.92], ...
        'EdgeColor', 'none', 'HandleVisibility', 'off');
  for k = 1:nm
    plot(a, D.q_deg, squeeze(D.norm_length(jnt,:,k)), '-', 'Color', cMus(k,:), 'LineWidth', LW);
  end
  yline(a, 1, ':', 'Color', [0.5 0.5 0.5]);
  xlim(a, [-120 120]);  xticks(a, -120:60:120);  ylim(a, [0.5 1.5]);
  grid(a, 'on');  box(a, 'on');  a.FontSize = fontsize;
  xlabel(a, sprintf('%s angle (deg)', jn{jnt}));
  if jnt == 1, ylabel(a, 'Fiber length (L_0)'); else, a.YTickLabel = []; end
  panel(a, lab{jnt});
end

% ---------------------------------------------------
% F: peak isometric force per muscle, each strength ratio (lambda = 1)
% ---------------------------------------------------
a = nexttile(tl, 4, [2 1]);  hold(a, 'on');
F = F_ref * cell2mat(SR')';                     % nm x 3
b = bar(a, 1:nm, F, 'grouped', 'EdgeColor', 'none');
for s = 1:3, b(s).FaceColor = cSR(s,:); end
xticks(a, 1:nm);  xticklabels(a, names);  xlim(a, [0.4 nm+0.6]);
grid(a, 'on');  box(a, 'on');  a.FontSize = fontsize;
ylabel(a, 'Peak force at \lambda = 1 (N)');
legend(a, SRlab, 'Location', 'northeast', 'Box', 'off', 'FontSize', fontsize-1);
panel(a, 'F');

% =========================================================================
% Local helpers
% =========================================================================
function xy = densify_path(P, C, R)
  % Polyline through the muscle's via/wrap points, drawing consecutive points
  % that both lie on the same wrapping cylinder as an arc on its surface
  % (shorter way round) instead of a chord through it.
  xy = P(1,:);
  for k = 1:size(P,1)-1
    p = P(k,:);  q = P(k+1,:);
    arc = [];
    for c = 1:size(C,1)
      if abs(norm(p - C(c,:)) - R(c)) < 1e-3 && abs(norm(q - C(c,:)) - R(c)) < 1e-3
        a0 = atan2(p(2)-C(c,2), p(1)-C(c,1));
        a1 = atan2(q(2)-C(c,2), q(1)-C(c,1));
        da = mod(a1 - a0 + pi, 2*pi) - pi;
        t  = linspace(0, 1, 30)';
        arc = C(c,:) + R(c)*[cos(a0 + da*t), sin(a0 + da*t)];
        break
      end
    end
    if isempty(arc), xy = [xy; q]; else, xy = [xy; arc(2:end,:)]; end %#ok<AGROW>
  end
end
