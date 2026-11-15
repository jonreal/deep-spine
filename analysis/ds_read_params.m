function [keys, vals] = ds_read_params(path)
    % Returns keys and vals as cell arrays of strings.
    % Both are empty ({}) when no custom.txt exists.
    path_to_parameters = fullfile(path, 'custom.txt');
    fid = fopen(path_to_parameters, 'r');
    if fid == -1
        keys = {};
        vals = {};
        return;
    end
    raw = fread(fid, inf, 'uint8=>char')';
    fclose(fid);
    p = jsondecode(raw);

    % Recursively extract all leaf parameter keys and values
    leaves = ds_flatten_struct(p);

    % Truncate each section of keys to 3 characters. Clamp the cut position to
    % the token length so parts shorter than 3 chars (e.g. "Ia", "II") are kept
    % whole instead of erroring in extractBefore.
    short_keys = cellfun(@(s) join( ...
        cellfun(@(p) extractBefore(p, min(strlength(p) + 1, 4)), split(s, "_"), ...
        'UniformOutput', false), "_"), ...
        leaves(:,1), 'UniformOutput', false);

    str1 = 'env_kwargs_relative_muscle_scale';
    str2 = 'env_kwargs_body_mass';

    i1 = find(strcmp(leaves(:,1), str1));
    i2 = find(strcmp(leaves(:,1), str2));

    if ~isempty(i1)
        val = leaves{i1, 2};
        tag = sprintf('[%g,%g,%g]', val(1), val(3), val(5));
        leaves(i1, 2) = {tag};
    end

    if ~isempty(i2)
        val = leaves{i2, 2};
        tag = sprintf('[%g,%g]', val(2), val(3));
        leaves(i2, 2) = {tag};
    end

    % Convert values to strings, remove log path entry
    value_strs = cellfun(@(v) string(v), leaves(:,2), 'UniformOutput', true);
    value_strs(1) = [];
    short_keys(1) = [];

    keys = cellstr(string(short_keys));
    vals = cellstr(string(value_strs));
end
