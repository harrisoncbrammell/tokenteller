from __future__ import annotations

from typing import Any


def stringify(value: Any) -> str:
    # keep floats short
    if isinstance(value, float):
        return f"{value:.6g}"
    # join list stuff
    if isinstance(value, list):
        return ", ".join(stringify(item) for item in value)
    # join dict stuff
    if isinstance(value, dict):
        return ", ".join(f"{key}={stringify(val)}" for key, val in value.items())
    return str(value)


def render_table(rows: list[dict[str, Any]]) -> str:
    # stop early if there is nothing
    if not rows:
        return "(no rows)"

    # keep the first column order
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
