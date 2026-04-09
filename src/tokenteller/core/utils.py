from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: serialize_value(val) for key, val in asdict(value).items()}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): serialize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_value(item) for item in value]
    return value


def stringify(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(stringify(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={stringify(val)}" for key, val in value.items())
    return str(value)


def render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    string_rows = [{column: stringify(row.get(column, "")) for column in columns} for row in rows]
    widths = {
        column: max(len(column), *(len(row[column]) for row in string_rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(row[column].ljust(widths[column]) for column in columns)
        for row in string_rows
    ]
    return "\n".join([header, divider, *body])
