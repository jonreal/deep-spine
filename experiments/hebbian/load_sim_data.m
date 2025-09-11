
base = './logs/exp9/trials/1756957634';

% Spinal/kinematic data
ncols = sum([1, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6]);
dtype = 'double';
fid = fopen([base,'/env.txt'], 'rb');
A = fread(fid,[ncols, Inf], dtype)';

S.qpos = A(:,(1:2) + 1);
S.qvel = A(:,(1:2) + 3);
S.len = A(:,(1:6) + 5);
S.vel = A(:,(1:6) + 5 + 6);
S.force = A(:,(1:6) + 5 + 12);
S.act_IIa = A(:,(1:6) + 5 + 18);
S.act_Ia = A(:,(1:6) + 5 + 24);
S.drive = A(:,(1:6) + 5 + 30);
S.act_MN = A(:,(1:6) + 5 + 36);
S.act_muscle = A(:,(1:6) + 5 + 42);
S.act = A(:,(1:6) + 5 + 48);
fclose(fid);

% Weight data
ncols = 36;
dtype = 'single';
fid = fopen([base,'/hebbian.txt'], 'rb');
A = fread(fid,[ncols, Inf], dtype)';
W = mat2cell(reshape(A',6,6,[]), 6, 6, ones(1, size(A,1)));
W = cellfun(@(w) w', squeeze(W)', 'UniformOutput', false);
fclose(fid);

