function [episode, time, muscle_strength, Wi, We] = ds_read_data(path)

  ncols = 39;  % episode, time, muscle_strength, W_Ia(36)

  % Detect binary vs ASCII: check if first byte is a printable digit
  fid = fopen(path, 'rb');
  first_byte = fread(fid, 1, 'uint8');
  fclose(fid);
  is_ascii = (first_byte >= 48 && first_byte <= 57);  % '0'-'9'

  if is_ascii
    data = readmatrix(path);
  else
    fid = fopen(path, 'rb');
    data = fread(fid, [ncols, Inf], 'double')';
    fclose(fid);
  end

  episode = data(:,1);
  time = data(:,2);
  muscle_strength = data(:,3);

  We = data(:, 4:end);
  We = mat2cell(reshape(We',6,6,[]), 6, 6, ones(1, size(We,1)));
  We = cellfun(@(w) w', squeeze(We)', 'UniformOutput', false);

  Wi = [];

end
