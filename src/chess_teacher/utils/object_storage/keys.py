from __future__ import annotations


def key_basename(key: str) -> str:
    """Return the last path segment of a POSIX storage key."""
    stripped = key.rstrip("/")
    if not stripped:
        return ""
    return stripped.rsplit("/", 1)[-1]


def relative_key_under(key: str, prefix: str) -> str:
    """Return ``key`` relative to ``prefix``, or its basename if outside the prefix."""
    prefix = prefix.strip("/")
    key = key.strip("/")
    if not prefix or key == prefix:
        return key
    head = f"{prefix}/"
    if key.startswith(head):
        return key[len(head) :]
    return key_basename(key)


def unique_key_variant(relative_key: str, unique: str) -> str:
    """Append ``unique`` before the extension (e.g. ``dir/a.jsonl`` -> ``dir/a_{unique}.jsonl``)."""
    basename = key_basename(relative_key)
    parent = relative_key[: -len(basename)].rstrip("/") if basename else ""
    if "." in basename:
        stem, suffix = basename.rsplit(".", 1)
        variant_name = f"{stem}_{unique}.{suffix}"
    else:
        variant_name = f"{basename}_{unique}" if basename else unique
    return f"{parent}/{variant_name}" if parent else variant_name


def sibling_temp_key(key: str) -> str:
    """Return a temporary sibling key (``file.jsonl`` -> ``file.jsonl.tmp``)."""
    basename = key_basename(key)
    parent = key[: -len(basename)].rstrip("/") if basename else ""
    temp_name = f"{basename}.tmp"
    return f"{parent}/{temp_name}" if parent else temp_name
