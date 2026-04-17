import re
import os
import json
import numpy as np
import torch


def resolve_path(path: str, context: dict):
    """Resolve dot-notation string (e.g. 'spine.net.W_Ia', 'learners[0]') to object reference."""
    parts = path.split(".")

    def get_attr_or_key(obj, name):
        """Get attribute or dict key."""
        if isinstance(obj, dict):
            return obj[name]
        return getattr(obj, name)

    def resolve_part(obj, part):
        # Handle indexing like "learners[0]" or "arr[2]"
        match = re.match(r'(\w+)\[(\d+)\]', part)
        if match:
            name, idx = match.groups()
            if obj is None:
                return context[name][int(idx)]
            return get_attr_or_key(obj, name)[int(idx)]
        # Regular attribute or dict key
        if obj is None:
            return context[part]
        return get_attr_or_key(obj, part)

    obj = None
    for part in parts:
        obj = resolve_part(obj, part)
    return obj


def resolve_kwargs(kwargs: dict, context: dict, keys_to_resolve: set = None):
    """Resolve string paths in kwargs dict. If keys_to_resolve is None, try all strings."""
    resolved = {}
    for k, v in kwargs.items():
        if isinstance(v, str) and (keys_to_resolve is None or k in keys_to_resolve):
            # Check if it looks like an object path (starts with a known context key)
            root = v.split(".")[0]
            # Strip index notation for context lookup (e.g. "learners[0]" -> "learners")
            root_base = re.match(r'(\w+)', root).group(1)
            if root_base in context:
                resolved[k] = resolve_path(v, context)
            else:
                resolved[k] = v
        else:
            resolved[k] = v
    return resolved


def recursive_update(d, u, *, strict=False):
    """Recursively update dict d with values from u."""
    changed = False

    # Special handling for lists of dictionaries (like hebbian_kwargs)
    if isinstance(d, list) and isinstance(u, list):
        # Make sure the lists are of the same length
        if len(d) != len(u):
            raise ValueError(f"[config error] Cannot update a list of length {len(d)} with a list of length {len(u)}")

        # Update each dictionary in the list
        for i in range(len(d)):
            if isinstance(d[i], dict) and isinstance(u[i], dict):
                changed |= recursive_update(d[i], u[i], strict=strict)
            else:
                if d[i] != u[i]:
                    d[i] = u[i]
                    changed = True
        return changed

    # Original code for dictionaries
    if isinstance(d, dict) and isinstance(u, dict):
        for k, v in u.items():
            if strict and k not in d:
                raise KeyError(f"[config error] unknown key '{k}' in override")
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                changed |= recursive_update(d[k], v, strict=strict)
            elif isinstance(v, list) and k in d and isinstance(d[k], list):
                # Handle nested lists (like in hebbian_kwargs)
                changed |= recursive_update(d[k], v, strict=strict)
            else:
                if k not in d or d[k] != v:
                    d[k] = v
                    changed = True
        return changed

    # Default case for non-dictionary, non-list values
    if d != u:
        d = u
        changed = True

    return changed


def convert_to_serializable(obj):
    """Convert numpy/torch objects to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    else:
        return obj


def parse_json_or_path(s):
    """Parse JSON string or load from file path."""
    if os.path.isfile(s):
        with open(s) as f:
            return json.load(f)
    return json.loads(s)
