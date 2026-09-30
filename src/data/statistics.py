import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from src.config import LABEL_NAMES, STATS_PATH


def compute_statistics(df):
    # Descriptive statistics of the cleaned data set, grouped by class
    
    df = df.assign(
        cls=df["label"].map(LABEL_NAMES),
        score=df["score"].round().astype(int),
        n_words=df["text"].str.split().str.len(),
        non_ascii=df["text"].str.contains(r"[^\x00-\x7F]"),
    )
    by_class = df.groupby("cls")

    overview = pd.Series(
        {
            "Reviews": len(df),
            "Hotels": df["hotel"].nunique(),
            "Cities": df["city"].nunique(),
            "Duplicate texts": df["text"].duplicated().sum(),
        }
    ).to_frame("Value")
    overview.index.name = "Metric"

    lengths = (
        df.groupby(["cls", "sentiment"])["n_words"]
        .agg(mean="mean", median="median", std="std", min="min", max="max")
        .round(1)
    )

    missing = pd.DataFrame(
        {
            "Empty upside": by_class["upside_missing"].sum(),
            "Empty downside": by_class["downside_missing"].sum(),
            "Non-ASCII chars": by_class["non_ascii"].sum(),
            "Non-ASCII share": by_class["non_ascii"].mean().round(3),
        }
    )

    return {
        "Overview": overview,
        "Class x sentiment": pd.crosstab(df["cls"], df["sentiment"], margins=True, margins_name="total"),
        "Review score by class and sentiment": pd.crosstab([df["cls"], df["sentiment"]], df["score"]),
        "Review length in words": lengths,
        "Missing parts and non-ASCII characters": missing,
    }


def _flat(table):
    """Turn index and columns into plain string headers and rows."""
    n_index = table.index.nlevels
    table = table.reset_index()
    table = table.rename(columns={"cls": "class"})
    headers = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in table.columns]
    rows = [[_fmt(v) for v in row] for row in table.itertuples(index=False)]
    return headers, rows, n_index


def _is_numeric_column(rows, i):
    try:
        for row in rows:
            float(row[i].rstrip("%"))
        return True
    except ValueError:
        return False


def _fmt(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def print_statistics(stats, title="Data statistics"):
    console = Console()
    console.rule(f"[bold]{title}")
    for name, table in stats.items():
        headers, rows, n_index = _flat(table)
        rich_table = Table(title=name, title_style="bold cyan", box=box.SIMPLE_HEAVY)
        for i, header in enumerate(headers):
            numeric = i >= n_index and _is_numeric_column(rows, i)
            rich_table.add_column(header, justify="right" if numeric else "left")
        for row in rows:
            rich_table.add_row(*row)
        console.print(rich_table)


def save_statistics(stats, path=STATS_PATH, title="Data statistics"):
    """Write all tables into one Markdown file (default: results/data_statistics.md)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for name, table in stats.items():
        headers, rows, n_index = _flat(table)
        lines += [f"## {name}", ""]
        lines.append("| " + " | ".join(headers) + " |")
        align = ["---:" if i >= n_index and _is_numeric_column(rows, i) else "---" for i in range(len(headers))]
        lines.append("|" + "|".join(align) + "|")
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
