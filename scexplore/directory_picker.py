"""Cross-platform directory picker and demo dataset generator for scExplore."""

from __future__ import annotations

from pathlib import Path
import platform
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def pick_directory_dialog(initial_dir: str | Path | None = None) -> str | None:
    """Open a native OS folder selection dialog and return the chosen path, or None if cancelled."""
    start_dir = str(Path(initial_dir or Path.cwd()).expanduser().resolve())
    system = platform.system()

    if system == "Darwin":
        # macOS native AppleScript choose folder dialog
        script = (
            f'POSIX path of (choose folder with prompt "Select directory containing single-cell data" '
            f'default location POSIX file "{start_dir}")'
        )
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass
        return None

    # Linux / Windows fallback using tkinter in a subprocess
    py_code = (
        "import tkinter as tk; from tkinter import filedialog; "
        "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True); "
        f"folder = filedialog.askdirectory(initialdir={start_dir!r}, title='Select directory containing single-cell data'); "
        "print(folder)"
    )
    try:
        proc = subprocess.run(
            ["python3", "-c", py_code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return None


def generate_demo_dataset(output_dir: str | Path) -> Path:
    """Create a realistic synthetic multi-sample merged PBMC single-cell dataset (.h5ad) for testing."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_merged_pbmc.h5ad"

    rng = np.random.default_rng(42)

    cell_types = ["CD4 T", "CD8 T", "B Cells", "NK Cells", "CD14 Monocytes", "Dendritic Cells"]
    samples = ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]
    sample_to_condition = {
        "Sample_1": "Control",
        "Sample_2": "Control",
        "Sample_3": "Stimulated",
        "Sample_4": "Stimulated",
    }
    sample_to_batch = {
        "Sample_1": "Batch_A",
        "Sample_2": "Batch_B",
        "Sample_3": "Batch_A",
        "Sample_4": "Batch_B",
    }

    marker_genes = {
        "CD4 T": ["CD3D", "CD3E", "IL7R", "CCR7", "CD4", "TCF7", "LEF1", "MAL"],
        "CD8 T": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMK", "CCL5", "CD2", "LCK"],
        "B Cells": ["MS4A1", "CD79A", "CD79B", "CD19", "HLA-DRA", "BANK1", "TCL1A", "CD74"],
        "NK Cells": ["GNLY", "NKG7", "PRF1", "GZMB", "GZMA", "KLRD1", "FGFBP2", "CST7"],
        "CD14 Monocytes": ["LYZ", "CD14", "S100A8", "S100A9", "FCN1", "LGALS3", "CTSS", "TYROBP"],
        "Dendritic Cells": ["FCER1A", "CST3", "CD1C", "CLEC10A", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DRB1"],
    }
    ifn_stimulated_genes = ["ISG15", "IFI6", "IFIT1", "IFIT3", "MX1", "OAS1", "STAT1", "CXCL10"]
    housekeeping_genes = [
        "ACTB", "GAPDH", "B2M", "RPL13A", "RPS18", "EEF1A1", "TPT1", "FTL", "FTH1",
        "MALAT1", "PTMA", "TMSB4X", "TMSB10", "PPIA", "RPLP0", "RPS6", "RPL10", "HSP90AA1",
    ]

    all_genes: list[str] = []
    for g_list in marker_genes.values():
        for g in g_list:
            if g not in all_genes:
                all_genes.append(g)
    for g in ifn_stimulated_genes + housekeeping_genes:
        if g not in all_genes:
            all_genes.append(g)

    # Pad to 120 genes total
    while len(all_genes) < 120:
        all_genes.append(f"GENE{len(all_genes) + 1}")

    n_cells = 600
    n_genes = len(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    cell_ct = rng.choice(cell_types, size=n_cells, p=[0.28, 0.18, 0.16, 0.12, 0.18, 0.08])
    cell_sample = rng.choice(samples, size=n_cells)
    cell_condition = np.array([sample_to_condition[s] for s in cell_sample])
    cell_batch = np.array([sample_to_batch[s] for s in cell_sample])

    counts = rng.poisson(lam=0.25, size=(n_cells, n_genes)).astype(np.float32)

    # Housekeeping baseline
    for g in housekeeping_genes:
        counts[:, gene_to_idx[g]] += rng.poisson(lam=8.0, size=n_cells)

    # Cell-type specific expression
    for ct, m_genes in marker_genes.items():
        mask = cell_ct == ct
        for g in m_genes:
            counts[mask, gene_to_idx[g]] += rng.poisson(lam=14.0, size=int(mask.sum()))

    # Condition-specific (Stimulated IFN response)
    stim_mask = cell_condition == "Stimulated"
    for g in ifn_stimulated_genes:
        counts[stim_mask, gene_to_idx[g]] += rng.poisson(lam=12.0, size=int(stim_mask.sum()))

    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(cell_sample),
            "condition": pd.Categorical(cell_condition),
            "cell_type": pd.Categorical(cell_ct),
            "batch": pd.Categorical(cell_batch),
        },
        index=[f"CELL_{i:04d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=pd.Index(all_genes, name="gene"))

    adata = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=obs,
        var=var,
        layers={"counts": sp.csr_matrix(counts.copy())},
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=25)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=42)
    sc.tl.umap(adata, random_state=42)

    adata.write_h5ad(out_path)
    return out_path
