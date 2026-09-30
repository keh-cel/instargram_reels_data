"""Turns whatever the user typed (a plain username, "@username", "#hashtag",
or a pasted instagram.com URL) into a normalized (source_type, value, label).
"""

import hashlib
import re
from urllib.parse import urlparse

VALID_MODES = {"auto", "account", "hashtag"}


def slugify(source_type: str, value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
    return f"{source_type}_{safe}_{digest}" if safe else f"{source_type}_{digest}"


def detect_source(query: str, mode: str = "auto") -> tuple:
    """Returns (source_type, value, label). Raises ValueError on bad input."""
    q = (query or "").strip()
    if not q:
        raise ValueError("계정 아이디, #해시태그 또는 인스타그램 URL을 입력하세요.")

    if mode not in VALID_MODES:
        mode = "auto"

    if mode == "account":
        value = q.lstrip("@").strip("/")
        return "account", value, f"@{value}"

    if mode == "hashtag":
        value = q.lstrip("#").strip()
        return "hashtag", value, f"#{value}"

    # auto-detect
    if q.startswith("#"):
        value = q[1:].strip()
        return "hashtag", value, f"#{value}"

    if "instagram.com" in q:
        parts = [p for p in urlparse(q).path.split("/") if p]
        if not parts:
            raise ValueError("URL에서 계정/해시태그를 찾을 수 없습니다.")
        if parts[0] == "explore" and "tags" in parts:
            value = parts[parts.index("tags") + 1]
            return "hashtag", value, f"#{value}"
        value = parts[0]
        return "account", value, f"@{value}"

    if q.startswith("@"):
        value = q[1:].strip()
        return "account", value, f"@{value}"

    return "account", q, f"@{q}"
