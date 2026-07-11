def parse_kwargs(value: str | None) -> dict[str, str]:
    """Parse comma separated key=value strings into a dict of strings."""
    if not value:
        return {}

    kwargs: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            k = item
            v = "True"
        else:
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
        if not k:
            raise ValueError("Key cannot be empty in key=value expression")
        kwargs[k] = v
    return kwargs


# Backwards compatibility alias
parse_key_value_string = parse_kwargs


def parse_scoped_kwargs(value: str):
    """Parse name:key=value into {name: {key: value}} preserving scopes."""

    if not value:
        return {}

    scoped: dict = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item or "=" not in item:
            raise ValueError("Scoped kwargs must look like name:key=value")
        scope_part, kv_part = item.split(":", 1)
        key, val = kv_part.split("=", 1)
        scope = scope_part.strip()
        key = key.strip()
        val = val.strip()
        if not scope or not key:
            raise ValueError("Scope and key cannot be empty")
        scoped.setdefault(scope, {})[key] = val

    return scoped


def parse_scoped_option(ctx, param, value):
    """Click callback wrapper around parse_scoped_kwargs."""

    del ctx, param  # unused in pure parsing
    return parse_scoped_kwargs(value)
