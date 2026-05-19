function exp2mat(path_to_experiments)
  % Generate list of experments
  dirs = dir([path_to_experiments,'/*']);
  id = {};
  tags_exsist = 0;

  for i = 1:length(dirs)
    name = dirs(i).name;
    if dirs(i).isdir && ~any(strcmp(name, {'.', '..', '.DS_Store'}))

      % Check if data file exists
      path = [path_to_experiments,'/',name];
      if exist([path,'/env.txt'], 'file')
          fprintf(['\nProcessing: ', path,'\n']);
      else
          fprintf(['\nSkipping: ', path,'\n']);
          continue;
      end

      % Spinal/kinematic data
      ncols = sum([1, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6]);
      dtype = 'double';
      fid = fopen([path,'/env.txt'], 'rb');
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
      fid = fopen([path,'/hebbian.txt'], 'rb');
      A = fread(fid,[ncols, Inf], dtype)';
      W = mat2cell(reshape(A',6,6,[]), 6, 6, ones(1, size(A,1)));
      W = cellfun(@(w) w', squeeze(W)', 'UniformOutput', false);
      fclose(fid);

      save([path,'/simulation_data.mat'],'S','W');
    end
  end
end
