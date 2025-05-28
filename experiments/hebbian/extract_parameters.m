function [key_str, val_str] = extract_parameters(path)
    % Load JSON file
    path_to_parameters = fullfile(path, 'custom.txt');
    fid = fopen(path_to_parameters, 'r');
    raw = fread(fid, inf, 'uint8=>char')';
    fclose(fid);
    p = jsondecode(raw);

    % Recursively extract all leaf parameter keys and values
    leaves = extract_leaves(p);

    % Truncate each section of keys to 3 characters
    short_keys = cellfun(@(s) join( ...
        cellfun(@(p) extractBefore(p, 4), split(s, "_"), ...
        'UniformOutput', false), "_"), ...
        leaves(:,1), 'UniformOutput', false);

    % Convert values to strings
    value_strs = cellfun(@(v) string(v), leaves(:,2), 'UniformOutput', true);

    % Remove log path
    value_strs(1) = [];
    short_keys(1) = [];

    % Combine into newline-separated strings
    key_str = strjoin(string(short_keys), newline);
    val_str = strjoin(string(value_strs), newline);
end
