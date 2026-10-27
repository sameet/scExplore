# 🧬 scExplore

**scExplore** is an interactive, browser-based single-cell RNA-Seq exploration and analysis tool built with **Python**, **Streamlit**, **Scanpy**, **PHATE**, and **Plotly**.

It is designed to run smoothly on any standard laptop or workstation and supports exploring individual or merged single-cell RNA-Seq experiments stored in `.rds` (Seurat v3/v4/v5 or Bioconductor `SingleCellExperiment`), `.h5` (10x Genomics HDF5 or AnnData HDF5), and `.h5ad` formats.

---

## ✨ Features

1. **Lightweight & Cross-Platform**: Works on any computer (macOS, Linux, Windows) with reasonable compute power using sparse matrices and WebGL-accelerated interactive Plotly scatter plots.
2. **Multi-Format Single-Cell Support (`.rds`, `.h5`, `.h5ad`)**:
   - **Seurat `.rds`** (both classic `Assay` v3/v4 and `Assay5` v5 layered objects) and **SingleCellExperiment `.rds`** via pure-Python `rds2py` (no R runtime required).
   - **10x Genomics `.h5`** and **AnnData `.h5ad` / `.h5`** via `scanpy` and `h5py`.
   - Seamlessly handles merged multi-sample experiments.
3. **Easy Directory & File Browser**:
   - Native OS folder selection dialog (**📁 Browse Folder**), quick subfolder/parent navigation, or direct path input.
   - Automatic discovery of `.rds`, `.h5`, and `.h5ad` files in the selected directory (with optional recursive scanning).
   - Built-in **🧪 Generate Demo** button to create a synthetic merged multi-sample PBMC dataset (`sample_data/demo_merged_pbmc.h5ad`) for immediate testing.
4. **On-Demand UMAP & PHATE Visualization**:
   - Displays precomputed **UMAP** and **PHATE** embeddings immediately if present in the loaded object, or computes them on-demand with a single click.
   - Supports single-embedding exploration or side-by-side **UMAP & PHATE** comparison.
5. **Single-Gene Expression Plotting**:
   - Search or type any gene symbol (case-insensitive) to overlay its expression across the selected dimensionality reduction (UMAP, PHATE, PCA, t-SNE, etc.), along with summary statistics across cell types.
6. **Multi-Gene Module Score Calculation**:
   - Select or paste multiple genes to compute a composite module score (using Scanpy's `score_genes`, equivalent to Seurat's `AddModuleScore`) and plot it across UMAP, PHATE, or any available reduction.
7. **Violin Plots by Sample with Custom Metadata Coloring**:
   - Plot single-gene expression distributions across **Samples** on the X-axis and color/group violins by any metadata column (e.g., condition, batch, cell type).
8. **Violin Plots by Condition & Annotated Cell Type**:
   - Dedicated side-by-side and combined grouped violin plots for inspecting gene expression across **Conditions** and **Annotated Cell Types**.
9. **Consistent Color Palettes & Customizable Gradients**:
   - On initial data load, persistent discrete color maps are automatically created for **Cell Annotations** (`Paired` palette), **Samples** (`Dark2` palette), and **Conditions** (`Accent` palette).
   - Continuous expression and module score plots initialize with the reverse `RdGy` (`RdGy_r`) gradient, with an interactive dropdown allowing users to choose from multiple gradients (`RdGy_r`, `RdGy`, `Viridis`, `Plasma`, `Magma`, `Inferno`, `Cividis`, `RdBu_r`, `YlOrRd`, `Blues`, `Reds`).

---

## 🚀 Quick Start & Installation

### Option 1: Using `uv` (Recommended & Fastest)

```bash
# Clone the repository
git clone <your-github-repo-url>
cd scExplore

# Create a virtual environment and install dependencies
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Launch scExplore in your browser
.venv/bin/streamlit run app.py
```

### Option 2: Using Standard `venv` + `pip`

```bash
# Clone the repository
git clone <your-github-repo-url>
cd scExplore

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch scExplore in your browser
streamlit run app.py
```

You can also install `scExplore` as an editable package and use the `scexplore` CLI command:

```bash
.venv/bin/pip install -e .
.venv/bin/scexplore
```

---

## 📖 Usage Guide

1. **Select a Directory**:
   - In the left sidebar, click **📁 Browse Folder** to open a native folder picker, navigate subfolders under **📂 Navigate Subfolders**, or paste a folder path directly.
   - To try the app right away without your own file, click **🧪 Generate Demo** in the sidebar.
2. **Load an Experiment**:
   - Choose your `.rds`, `.h5`, or `.h5ad` file from the dropdown and click **🚀 Load Dataset**.
   - Map your dataset's metadata columns for **Sample**, **Condition**, and **Annotated Cell Type** in section **3. Metadata Mapping** (auto-detected by default).
3. **Explore Tabs**:
   - **🗺️ UMAP & PHATE Explorer**: Color cells by metadata on UMAP or PHATE (single view or side-by-side). Click **⚡ Compute UMAP Now** or **⚡ Compute PHATE Now** if an embedding is not yet present.
   - **🔍 Single-Gene Expression**: Plot expression of a gene across your chosen dimensionality reduction.
   - **🧮 Multi-Gene Module Score**: Enter or select a set of genes, click **🧮 Calculate Module Score**, and visualize the resulting module score on UMAP or PHATE.
   - **🎻 Violin Plots (Sample, Condition & Cell Type)**: Plot violin distributions for any gene across samples (with selectable metadata coloring), conditions, and annotated cell types.

---

## 🗂️ Project Structure

```text
scExplore/
├── app.py                        # Main Streamlit browser application
├── scexplore/
│   ├── __init__.py               # Package metadata
│   ├── analysis.py               # UMAP, PHATE, gene lookup & module score logic
│   ├── cli.py                    # CLI entry point (`scexplore`)
│   ├── data_loader.py            # Unified loader for .rds (Seurat/SCE), .h5, and .h5ad
│   ├── directory_picker.py       # Native folder dialog & synthetic demo dataset generator
│   └── plotting.py               # Interactive Plotly scatter and violin plot builders
├── tests/
│   └── test_scexplore.py         # End-to-end test suite
├── pyproject.toml                # Python package configuration & dependencies
├── requirements.txt              # Pip installable requirements list
├── .gitignore                    # Git ignore rules
└── README.md                     # Documentation
```

---

## 🧪 Running Tests

```bash
PYTHONPATH=. .venv/bin/python tests/test_scexplore.py
```
