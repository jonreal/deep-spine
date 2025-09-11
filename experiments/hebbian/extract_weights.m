function [Wi, We, t, episode] = extract_weights(path)
  I = readmatrix([path,'/W_n.txt']);
  E = readmatrix([path,'/W_p.txt']);
  episode = I(:,1);
  t = I(:,2);
  Wi = I(:,3:end);
  We = E(:,3:end);

  We = mat2cell(reshape(We',6,6,[]), 6, 6, ones(1, size(We,1)));
  We = cellfun(@(w) w', squeeze(We)', 'UniformOutput', false);

  Wi = mat2cell(reshape(Wi',6,6,[]), 6, 6, ones(1, size(Wi,1)));
  Wi = cellfun(@(w) w', squeeze(Wi)', 'UniformOutput', false);

end
