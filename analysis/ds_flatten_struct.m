function leaves = ds_flatten_struct(s, prefix)
    if nargin < 2
        prefix = "";
    end
    leaves = {};
    
    if isstruct(s)
        fn = fieldnames(s);
        for i = 1:numel(fn)
            name = fn{i};
            value = s.(name);
            new_prefix = prefix + name;
            if ~isempty(prefix)
                new_prefix = new_prefix + "_";
            end
            leaves = [leaves; ds_flatten_struct(value, new_prefix)];
        end

    elseif iscell(s)
        for i = 1:numel(s)
            value = s{i};
            new_prefix = prefix + string(i) + "_";
            leaves = [leaves; ds_flatten_struct(value, new_prefix)];
        end

    else
        % primitive value
        key = char(regexprep(prefix, '_$', ''));  % remove trailing underscore
        leaves = [{key}, {s}];
    end
end
