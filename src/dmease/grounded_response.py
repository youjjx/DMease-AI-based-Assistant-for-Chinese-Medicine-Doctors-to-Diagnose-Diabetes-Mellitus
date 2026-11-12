"""Explicit, auditable formatting tolerance without inventing JSON contents."""
import json
import re

from .local_extraction import validate_output


def parse_grounded_response(raw: str, source: str, canonical_names: set[str]) -> dict:
    strict = validate_output(raw, source, canonical_names)
    if not strict["invalid_json"]:
        return {**strict, "format_adjustment": "none"}
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.S)
    if fence:
        parsed = validate_output(fence.group(1), source, canonical_names)
        if not parsed["invalid_json"]:
            return {**parsed, "format_adjustment": "outer_json_fence"}
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except (ValueError, TypeError):
        return {**strict, "format_adjustment": "unrecoverable"}
    # Accept exactly one serialization artifact after a complete object. Never
    # discard arbitrary prose, another object, or a truncated second item.
    if isinstance(payload, dict) and text[end:].strip() == '"':
        parsed = validate_output(text[:end], source, canonical_names)
        if not parsed["invalid_json"]:
            return {**parsed, "format_adjustment": "trailing_quote_after_complete_object"}
    return {**strict, "format_adjustment": "unrecoverable"}
