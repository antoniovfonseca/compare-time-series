import nbformat as nbf
from pathlib import Path


def load_metadata(target_path: Path) -> dict:
    if not target_path.exists():
        return {}

    with target_path.open("r", encoding="utf-8") as handle:
        existing_nb = nbf.read(handle, as_version=4)

    return existing_nb.get("metadata", {})


def main() -> None:
    target_path = Path("notebook-time-series-comparison.ipynb")
    metadata = load_metadata(target_path)

    notebook = nbf.v4.new_notebook(metadata=metadata)
    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            """# Comparison of Two Time Series of Maps 0.1

*This streamlined notebook focuses on the statistical summaries and charts.
Provide your inputs in the first cells, run the calculations, and then generate
the figures directly from the CSV files exported by the workflow.*"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Environment Setup"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """# =============================================================================
# 1. Load the libraries
# =============================================================================
# Keep only the packages required for calculations and plotting.
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.ticker import FuncFormatter"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """### 1.1 Provide file paths and analysis parameters

*Edit the next cell so it matches your folders, time points, and NoData value before running the rest of the notebook.*"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """# =============================================================================
# 1. Define the analysis inputs (edit this cell)
# =============================================================================
# Provide the folders containing the raster time series for X and Y.
path_series_x = Path(
    r"C:\\Users\\AntFonseca\\github\\1.INPUT\\compare-time-series\\PIE\\pixelbased"
)
path_series_y = Path(
    r"C:\\Users\\AntFonseca\\github\\1.INPUT\\compare-time-series\\PIE\\objectbased"
)

# List the time points (years) to analyze.
time_points = [
    2010,
    2012,
    2014,
    2016,
    2018,
    2021
]

# Set a short class name used in filenames.
class_name = "PIE"

# Indicate the NoData value stored in the rasters.
nodata_value = 255

# Choose where the outputs will be stored.
output_path = Path(
    rf"C:\\Users\\AntFonseca\\github\\2.OUTPUT\\{class_name}"
)
output_path.mkdir(
    parents=True,
    exist_ok=True
)

# Example alternatives (uncomment and adjust as needed).
# path_series_x = Path(r"C:\\Users\\AntFonseca\\github\\compare-time-series\\input\\pieBasZoom2")
# path_series_y = Path(r"C:\\Users\\AntFonseca\\github\\compare-time-series\\input\\pieObjZoom2")
# time_points = [2010, 2012, 2014, 2016, 2018, 2021]
# class_name = "PIE"
# nodata_value = 255

# path_series_x = Path(r"C:\\Users\\AntFonseca\\github\\1.INPUT\\compare-time-series\\collection6")
# path_series_y = Path(r"C:\\Users\\AntFonseca\\github\\1.INPUT\\compare-time-series\\collection8")
# time_points = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
# class_name = "savanna"
# nodata_value = 255

# path_series_x = Path(r"C:\\Users\\AntFonseca\\github\\1.INPUT\\compare-time-series\\soilCol1")
# path_series_y = Path(r"C:\\Users\\AntFonseca\\github\\1.INPUT\\compare-time-series\\soilCol2")
# time_points = [1990, 2000, 2010]
# class_name = "soil"
# nodata_value = -32768

print("? Parameters successfully defined.")"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Helper Functions"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """# =============================================================================
# 2. Helper functions (no need to edit)
# =============================================================================
# The cache avoids reading the same rasters multiple times.
raster_arrays = {}
"""
        )
    )
