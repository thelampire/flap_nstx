#!/usr/bin/env python3
"""
dwscope_to_jscope.py

Convert old DWScope-style scope config files into JScope configuration syntax.

The output is written in canonical JScope style:
    Scope.key: value

Usage:
    python3 dwscope_to_jscope.py old_scope.conf new_scope.jscp
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


# Optional normalization for known aliases in hand-edited files.
# Most real DWScope files already use Scope.* keys and pass through unchanged.
EXACT_KEY_MAP: Dict[str, str] = {
    "title": "Scope.title",
    "icon_name": "Scope.icon_name",
    "columns": "Scope.columns",
}

# Pattern-based aliases for older/alternate notations.
PATTERN_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^scope\.(.+)$"), r"Scope.\1"),
    (re.compile(r"^plot_(\d+)_(\d+)\.(.+)$", re.I), r"Scope.plot_\1_\2.\3"),
    (re.compile(r"^global_(\d+)_(\d+)\.(.+)$", re.I), r"Scope.global_\1_\2.\3"),
    (re.compile(r"^rows_in_column_(\d+)$", re.I), r"Scope.rows_in_column_\1"),
]


JSCOPE_SERVER_DEFAULTS: List[str] = [
    "jScope.default_server = 9",
    "jScope.data_server_9.name = PPPL NSTX",
    "jScope.data_server_9.class = MdsDataProvider",
    "jScope.data_server_9.argument = skylark.pppl.gov:8501",
]


def parse_properties(text: str) -> Dict[str, str]:
    """
    Parse loose key/value file.
    Supports lines like:
      key=value
      key: value
    Ignores blank lines and #/; comments.
    """
    data: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue

        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            # If a line has no delimiter, skip it safely.
            continue

        data[key.strip()] = value.strip()
    return data


def normalize_value(v: str) -> str:
    """Small value normalizations for stable output."""
    lv = v.strip().lower()
    if lv == "true":
        return "true"
    if lv == "false":
        return "false"
    return v.strip()


CURRENT_SHOT_RE = re.compile(r'^current_shot\("[^"]+"\)$', re.I)


def remap_key(old_key: str) -> str:
    key = old_key.strip()
    lk = key.lower()

    if lk in EXACT_KEY_MAP:
        return EXACT_KEY_MAP[lk]

    for pattern, replacement in PATTERN_MAP:
        if pattern.match(key):
            return pattern.sub(replacement, key)

    # If already a Scope.* key, pass through untouched.
    if key.startswith("Scope."):
        return key

    # Conservative fallback: keep key text, just add Scope. prefix.
    return f"Scope.{key}"


def convert(dwscope_kv: Dict[str, str], keep_current_shot: bool = False) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in dwscope_kv.items():
        new_key = remap_key(k)
        new_value = normalize_value(v)

        # JScope can throw "undefined data class for not connected" when
        # evaluating current_shot(...) before an active MDSplus connection.
        # Use a safe numeric placeholder unless explicitly requested otherwise.
        if not keep_current_shot and new_key.lower().endswith(".shot") and CURRENT_SHOT_RE.match(new_value):
            new_value = "0"

        out[new_key] = new_value
    return out


def write_properties(data: Dict[str, str], output_path: Path) -> None:
    lines = []
    lines.extend(JSCOPE_SERVER_DEFAULTS)
    lines.append("")
    for k, v in data.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DWScope config to JScope config")
    parser.add_argument("input", type=Path, help="Path to old DWScope file")
    parser.add_argument("output", type=Path, help="Path to new JScope output file (.jscp)")
    parser.add_argument(
        "--keep-current-shot",
        action="store_true",
        help="Keep current_shot(\"tree\") expressions in *.shot fields instead of replacing with 0",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    src_text = args.input.read_text(encoding="utf-8", errors="replace")
    src_data = parse_properties(src_text)
    new_data = convert(src_data, keep_current_shot=args.keep_current_shot)

    if "Scope.columns" not in new_data:
        raise SystemExit(
            "Converted output is missing required key 'Scope.columns'. "
            "Check input syntax or extend key mapping."
        )

    output_path = args.output
    if output_path.suffix.lower() != ".jscp":
        output_path = output_path.with_suffix(".jscp")
        print(f"Adjusted output extension to .jscp: {output_path}")

    write_properties(new_data, output_path)

    print(f"Converted {len(src_data)} entries -> {len(new_data)} entries")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
