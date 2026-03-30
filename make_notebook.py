#!/usr/bin/env python3
"""
Converts pipeline_notebook.py into a proper Jupyter .ipynb file.
Run: python3 make_notebook.py
"""
import json, re

CELLS_RAW = open("pipeline_notebook.py").read()

# Split on cell markers
cell_blocks = re.split(r"# ─{70,}\n# CELL \d+:.*\n# ─{70,}\n", CELLS_RAW)

# First block is the module docstring
header_block = cell_blocks[0]
content_blocks = cell_blocks[1:]

cells = []

# Header cell (markdown)
header_md = []
for line in header_block.strip().splitlines():
    line = line.lstrip("# ")
    header_md.append(line)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [l + "\n" for l in header_md],
})

# Content cells
cell_titles = re.findall(r"# CELL \d+: (.+)", CELLS_RAW)

for i, (title, block) in enumerate(zip(cell_titles, content_blocks)):
    lines = block.strip().splitlines()
    # Extract leading comment block as markdown header
    comment_lines = []
    code_lines = []
    in_comment = True
    for line in lines:
        if in_comment and (line.startswith("# ") or line == "#"):
            comment_lines.append(line.lstrip("# "))
        else:
            in_comment = False
            code_lines.append(line)

    if comment_lines:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {title}\n"] + [l + "\n" for l in comment_lines],
        })

    if code_lines:
        src = [l + "\n" for l in code_lines]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src,
        })

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

with open("pipeline_notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Created: pipeline_notebook.ipynb")
