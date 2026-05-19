figure;

ncol = 2
nrow = sum(arrayfun(@(i) size(D{i},2),1:numel(D)));
xlab = 0.1;
ylab = 0.1;
ysp = 0.005;
xsp = 0.02;
w = (1 - 2*ylab - (ncol-1)*xsp)/ncol;
h = (1 - 2*xlab - (nrow-1)*ysp)/nrow;

c1 = [0.2 0 0];
c2 = [0 0.2 0];
c3 = [0.2 0.2 0.2];

C = {[c1;c2;c1;c2],repmat(c3,6,1)};

xs = ylab;
ys = 0.9 - h;

for k=1:numel(D)
  D_k = D{k};
  C_k = C{k};
  for j=1:size(D_k,2)
    pos = [xs,ys,w,h];
    axes('Position',pos);
    plot(D_k(:,j),'k','linewidth',0.1,'Color',C_k(j,:));
    axis off;

    ys = ys - h - ysp;
  end
end

nweights = size(W_t,2);
h = 0.8;
xs = ylab + w + xsp;
ys = 0.9 - h;

pos = [xs,ys,w,h];
ax = axes('Position',pos);
for i=1:size(W_t,2)
  plot(gradient(W_t(:,i)),'k','linewidth',0.1,'Color',C_w(i,:)); hold all;
end

ax.Color = 'none';


