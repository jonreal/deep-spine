function [episode, time, muscle_strength, Wi, We] = extract_data(path)

  data = readmatrix(path);

  episode = data(:,1);
  time = data(:,2);
  muscle_strength = data(:,3);

  Wi = data(:,(1:36) + 3);
  We = data(:,(1:36) + 39);

  We = mat2cell(reshape(We',6,6,[]), 6, 6, ones(1, size(We,1)));
  We = cellfun(@(w) w', squeeze(We)', 'UniformOutput', false);

  Wi = mat2cell(reshape(Wi',6,6,[]), 6, 6, ones(1, size(Wi,1)));
  Wi = cellfun(@(w) w', squeeze(Wi)', 'UniformOutput', false);

end
