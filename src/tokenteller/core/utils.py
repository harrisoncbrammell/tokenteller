from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def serialize_value(value: Any) -> Any:
    # Flatten dataclasses into plain nested Python objects.
    if is_dataclass(value):
        return {key: serialize_value(val) for key, val in asdict(value).items()}
    # Respect helper objects that already know how to serialize themselves.
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    # Recurse through dictionaries.
    if isinstance(value, dict):
        return {str(key): serialize_value(val) for key, val in value.items()}
    # Recurse through common sequence types.
    if isinstance(value, (list, tuple, set)):
        return [serialize_value(item) for item in value]
    # Return primitive values unchanged.
    return value


def stringify(value: Any) -> str:
    # Format floats compactly for text tables.
    if isinstance(value, float):
        return f"{value:.6g}"
    # Join list values into one readable string.
    if isinstance(value, list):
        return ", ".join(stringify(item) for item in value)
    # Join dictionary values into a compact key=value string.
    if isinstance(value, dict):
        return ", ".join(f"{key}={stringify(val)}" for key, val in value.items())
    # Fall back to the normal string conversion.
    return str(value)


def render_table(rows: list[dict[str, Any]]) -> str:
    # Handle the empty case without building a full table.
    if not rows:
        return "(no rows)"

    # Preserve the order in which columns first appear.
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    # Convert all cells to strings before measuring widths.
    string_rows = [{column: stringify(row.get(column, "")) for column in columns} for row in rows]
    # Compute the width of each column from the header and body.
    widths = {
        column: max(len(column), *(len(row[column]) for row in string_rows))
        for column in columns
    }
    # Build the header row and divider line.
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    # Build the table body row by row.
    body = [
        " | ".join(row[column].ljust(widths[column]) for column in columns)
        for row in string_rows
    ]
    # Join the pieces into one plain text table.
    return "\n".join([header, divider, *body])
