import pandas as pd
import numpy as np

def get_describe_stats(df):
    """
    Returns a dict of describe() stats for each numeric column.
    """
    describe = df.describe().to_dict()
    stats = {}
    for col, metrics in describe.items():
        stats[col] = {
            "count": metrics.get("count"),
            "mean": metrics.get("mean"),
            "std": metrics.get("std"),
            "min": metrics.get("min"),
            "25%": metrics.get("25%"),
            "50%": metrics.get("50%"),
            "75%": metrics.get("75%"),
            "max": metrics.get("max"),
        }
    return stats