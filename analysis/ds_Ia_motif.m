function motif = ds_Ia_motifs(W,opt)
  arguments
    W
    opt.show_basis = false
  end

  [m,n] = size(W{1});
  assert(m == n,'Dim mismatch');

  S_template = [ ...
    1, 0, zeros(1,n-2);
    0, 1, zeros(1,n-2); ...
    zeros(n-2,n)];
  bS = ...
        arrayfun(@(i) ...
          arrayfun(@(j) ...
            reshape(circshift(S_template, [i j]),[],1), ...
            [0:2:(n-2)],'UniformOutput',0), ...
          [0:2:(n-2)],'UniformOutput',0);

  A_template = [ ...
    0, 1, zeros(1,n-2);
    1, 0, zeros(1,n-2); ...
    zeros(n-2,n)];
  bA = ...
        arrayfun(@(i) ...
          arrayfun(@(j) ...
            reshape(circshift(A_template, [i j]),[],1), ...
            [0:2:(n-2)],'UniformOutput',0), ...
          [0:2:(n-2)],'UniformOutput',0);
  bR = cell2mat(arrayfun(@(i) bS{i}{i}, 1:(n/2), 'UniformOutput',0));

  for i=1:(n/2)
    bS{i}(i) = [];
  end

  bS_mat = cell2mat(cellfun(@(x) [x{:}], bS, 'UniformOutput', false));
  bA_mat = cell2mat(cellfun(@(x) [x{:}], bA, 'UniformOutput', false));
  B = [bR, bS_mat, bA_mat];

  Binv = pinv(B);
  c = [cellfun(@(Wq) Binv*Wq(:), W,'UniformOutput',0)];
  residual = cell2mat(...
    [cellfun(@(Wq,cq) norm(Wq(:) - B*cq), W,c,'UniformOutput',0)])';
  c = cell2mat(c)';

  % To level is the symmetry level (flexor and extensor have single gain)
  % ---------------------------------------------------------------------------
  np = n/2;
  motif.r = c(:,1:np);
  motif.s = c(:,(1:(2*np))+np);
  motif.a = c(:,(1:(3*np))+3*np);
  motif.res = residual;
  motif.mask = logical(B);
  motif.score = ( ...
    mean(motif.r,2).*mean(motif.s,2).*(1 - mean(motif.a,2)));

  % Partion 'Pool' level
  % ---------------------------------------------------------------------------
  rp = motif.r;
  sp = cell2mat(arrayfun(@(i) ...
              mean(motif.s(:,1:(np-1) + (i-1)*(np-1)), 2), ...
              1:np, 'UniformOutput', 0));
  ap = cell2mat(arrayfun(@(i) ...
              mean(motif.a(:, (1:np) + (i-1)*np), 2), ...
              1:np, 'uniformoutput', 0));
  cp = [rp, sp, ap];
  Brp = B(:,1:np);
  Bsp = cell2mat(arrayfun(@(i) ...
              sum(B(:, (1:(np-1)) + np + (i-1)*(np-1)), 2), ...
              1:np, 'UniformOutput', 0));
  Bap = cell2mat(arrayfun(@(i) ...
              sum(B(:, (1:np) + np + np*(np-1) + (i-1)*np), 2), ...
              1:np, 'uniformoutput', 0));
  Bp = [Brp, Bsp, Bap];

  residual_pool = cell2mat(arrayfun(@(i) norm(W{i}(:) - Bp * cp(i,:)'), ...
                    1:size(cp, 1),'UniformOutput', 0))';

  motif.rp = rp;
  motif.sp = sp;
  motif.ap = ap;
  motif.resp = residual_pool;
  motif.maskp= logical(Bp);

  % Partion 'Global' level
  % ---------------------------------------------------------------------------
  rg = mean(rp,2);
  sg = mean(sp,2);
  ag = mean(ap,2);
  cg = [rg, sg, ag];

  Brg = sum(Brp,2);
  Bsg = sum(Bsp,2);
  Bag = sum(Bap,2);
  Bg = [Brg, Bsg, Bag];

  residual_global = cell2mat(...
      arrayfun(@(i) ...
        norm(W{i}(:) - Bg * cg(i,:)'), ...
      1:size(cg, 1), ...
      'UniformOutput', 0))';

  motif.rg = rg;
  motif.sg = sg;
  motif.ag = ag;
  motif.resg = residual_global;
  motif.maskg = logical(Bg);

  if opt.show_basis

    figure;
    n = size(bR,2);
    for i=1:n
      subplot(1,n,i)
      imagesc(reshape(bR(:,i),6,6))
    end

    figure;
      n = numel(bS)
      for i=1:n
        for j=1:(n-1)
          subplot(1,n*(n-1), j + (i-1)*(n-1));
          imagesc(reshape(bS{i}{j},6,6));
          title(['(',num2str(i),num2str(j),')']);
        end
      end

    figure;
      n = numel(bA)
      for i=1:n
        for j=1:n
          subplot(n,n, j + (i-1)*n);
          imagesc(reshape(bA{i}{j},6,6));
          title(['(',num2str(i),num2str(j),')']);
        end
      end
  end

end
