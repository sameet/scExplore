"""Data loading utilities for scExplore (.h5ad, 10x .h5, Seurat .rds, SCE .rds)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import rds2py
from rds2py.generics import _dispatcher
import scanpy as sc
import scipy.sparse as sp


SUPPORTED_EXTENSIONS = (".h5ad", ".h5", ".rds")


def list_sc_files(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Return sorted single-cell data files (.h5ad, .h5, .rds) in a directory."""
    base = Path(directory).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in base.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def _extract_matrix_and_dimnames(
    mat_dict: dict[str, Any],
) -> tuple[sp.spmatrix | np.ndarray, list[str] | None, list[str] | None]:
    """Extract matrix and (rownames, colnames) from an rds2py parsed matrix object."""
    dispatched = _dispatcher(mat_dict)
    rownames: list[str] | None = None
    colnames: list[str] | None = None

    if hasattr(dispatched, "matrix"):
        mat = dispatched.matrix
        if getattr(dispatched, "dimnames", None) and len(dispatched.dimnames) == 2:
            rownames = (
                list(dispatched.dimnames[0])
                if dispatched.dimnames[0] is not None
                else None
            )
            colnames = (
                list(dispatched.dimnames[1])
                if dispatched.dimnames[1] is not None
                else None
            )
    else:
        mat = dispatched

    # Fallback: check raw attributes['Dimnames'] or attributes['dimnames']
    attrs = mat_dict.get("attributes") or {}
    dimnames_raw = attrs.get("Dimnames") or attrs.get("dimnames")
    if dimnames_raw and isinstance(dimnames_raw, dict):
        dn_data = dimnames_raw.get("data", [])
        if len(dn_data) >= 2:
            if rownames is None and isinstance(dn_data[0], dict) and dn_data[0].get("data") is not None:
                rownames = [str(x) for x in dn_data[0]["data"]]
            if colnames is None and isinstance(dn_data[1], dict) and dn_data[1].get("data") is not None:
                colnames = [str(x) for x in dn_data[1]["data"]]

    return mat, rownames, colnames


def _extract_logmap_names(logmap_dict: dict[str, Any]) -> list[str] | None:
    """Extract row names from a Seurat v5 LogMap S4/boolean_vector object."""
    attrs = logmap_dict.get("attributes") or {}
    dimnames = attrs.get("dimnames") or attrs.get("Dimnames")
    if dimnames and isinstance(dimnames, dict):
        dn_data = dimnames.get("data", [])
        if len(dn_data) >= 1 and isinstance(dn_data[0], dict) and dn_data[0].get("data") is not None:
            return [str(x) for x in dn_data[0]["data"]]
    return None


def _parse_seurat_rds(raw_rds: dict[str, Any]) -> ad.AnnData:
    """Convert a parsed Seurat (v3/v4 Assay or v5 Assay5) RDS dict into AnnData."""
    attrs = raw_rds["attributes"]

    # 1. Cell metadata (meta.data)
    meta_bf = _dispatcher(attrs["meta.data"])
    obs_df = meta_bf.to_pandas()
    if "rownames" in obs_df.columns:
        obs_df.index = obs_df["rownames"].astype(str)
        obs_df = obs_df.drop(columns=["rownames"])
    else:
        obs_df.index = [str(i) for i in obs_df.index]

    # Also capture active.ident if present and not already in obs_df
    if "active.ident" in attrs and attrs["active.ident"].get("type") != "null":
        try:
            active_ident = _dispatcher(attrs["active.ident"])
            if hasattr(active_ident, "__len__") and len(active_ident) == len(obs_df):
                obs_df["seurat_active_ident"] = [str(x) for x in active_ident]
        except Exception:
            pass

    # 2. Select active assay (or default to RNA / first assay)
    assays_dict = attrs["assays"]
    assay_names = [str(x) for x in assays_dict["attributes"]["names"]["data"]]
    active_assay_name = "RNA"
    if "active.assay" in attrs and attrs["active.assay"].get("data"):
        active_assay_name = str(attrs["active.assay"]["data"][0])
    if active_assay_name not in assay_names:
        active_assay_name = "RNA" if "RNA" in assay_names else assay_names[0]

    assay_idx = assay_names.index(active_assay_name)
    assay_obj = assays_dict["data"][assay_idx]
    assay_class = assay_obj.get("class_name", "Assay")
    assay_attrs = assay_obj["attributes"]

    layers: dict[str, Any] = {}
    X_mat = None
    gene_names: list[str] | None = None
    cell_names: list[str] | None = None

    if assay_class == "Assay5" or "layers" in assay_attrs:
        # Seurat v5 Assay5 structure
        layer_list = assay_attrs["layers"]
        layer_names = [str(x) for x in layer_list["attributes"]["names"]["data"]]
        feat_names_all = _extract_logmap_names(assay_attrs.get("features", {}))
        cell_names_all = _extract_logmap_names(assay_attrs.get("cells", {}))

        # Prefer 'data' (log-normalized) for X, fallback to 'counts'
        for lname, ldict in zip(layer_names, layer_list["data"]):
            mat, rnames, cnames = _extract_matrix_and_dimnames(ldict)
            if mat is None or (hasattr(mat, "shape") and 0 in mat.shape):
                continue
            # Transpose from (genes, cells) to (cells, genes)
            mat_t = mat.T
            if sp.issparse(mat_t):
                mat_t = mat_t.tocsr()
            else:
                mat_t = np.asarray(mat_t)
            layers[lname] = mat_t
            if gene_names is None:
                gene_names = rnames or feat_names_all
            if cell_names is None:
                cell_names = cnames or cell_names_all

        if "data" in layers:
            X_mat = layers["data"]
        elif "counts" in layers:
            X_mat = layers["counts"]
        elif layers:
            X_mat = next(iter(layers.values()))
    else:
        # Seurat v3 / v4 Assay structure
        for slot_name in ("data", "counts"):
            if slot_name in assay_attrs and assay_attrs[slot_name].get("type") != "null":
                mat, rnames, cnames = _extract_matrix_and_dimnames(assay_attrs[slot_name])
                if mat is not None and hasattr(mat, "shape") and 0 not in mat.shape:
                    mat_t = mat.T
                    if sp.issparse(mat_t):
                        mat_t = mat_t.tocsr()
                    else:
                        mat_t = np.asarray(mat_t)
                    layers[slot_name] = mat_t
                    if gene_names is None and rnames is not None:
                        gene_names = rnames
                    if cell_names is None and cnames is not None:
                        cell_names = cnames

        if "data" in layers:
            X_mat = layers["data"]
        elif "counts" in layers:
            X_mat = layers["counts"]

    if X_mat is None:
        raise ValueError(f"Could not extract expression matrix from Seurat assay '{active_assay_name}'.")

    n_cells, n_genes = X_mat.shape
    if gene_names is None or len(gene_names) != n_genes:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
    if cell_names is None or len(cell_names) != n_cells:
        cell_names = list(obs_df.index[:n_cells])

    # Align obs_df to cell_names
    if len(obs_df) == n_cells:
        obs_df.index = pd.Index(cell_names, name="cell_id")
    else:
        obs_df = obs_df.reindex(cell_names)

    var_df = pd.DataFrame(index=pd.Index(gene_names, name="gene"))

    # Match layers with matching shape (n_cells, n_genes)
    valid_layers = {
        k: v for k, v in layers.items() if v.shape == (n_cells, n_genes)
    }

    adata = ad.AnnData(X=X_mat, obs=obs_df, var=var_df, layers=valid_layers)

    # Preserve raw counts in adata.raw if available
    if "counts" in valid_layers and "data" in valid_layers:
        adata.layers["counts"] = valid_layers["counts"]

    # 3. Extract dimensional reductions (UMAP, PHATE, PCA, tSNE, etc.)
    if "reductions" in attrs and attrs["reductions"].get("type") != "null":
        red_dict = attrs["reductions"]
        red_attrs = red_dict.get("attributes") or {}
        if "names" in red_attrs and red_attrs["names"].get("data") is not None:
            red_names = [str(x) for x in red_attrs["names"]["data"]]
            for rname, robj in zip(red_names, red_dict["data"]):
                try:
                    emb_dict = robj["attributes"]["cell.embeddings"]
                    emb_mat, r_cells, _ = _extract_matrix_and_dimnames(emb_dict)
                    emb_arr = np.asarray(emb_mat, dtype=np.float32)
                    if emb_arr.shape[0] == n_cells:
                        key = f"X_{rname.lower()}" if not rname.lower().startswith("x_") else rname.lower()
                        adata.obsm[key] = emb_arr
                except Exception:
                    continue

    return adata


def _parse_sce_rds(raw_rds: dict[str, Any], path: str | Path) -> ad.AnnData:
    """Convert a SingleCellExperiment or SummarizedExperiment RDS into AnnData."""
    sce = rds2py.read_rds(str(path))
    attrs = raw_rds.get("attributes", {})

    # Extract assays
    assay_names = list(sce.get_assay_names())
    if not assay_names:
        raise ValueError("SingleCellExperiment object contains no assays.")

    layers: dict[str, Any] = {}
    for aname in assay_names:
        mat = sce.get_assay(aname)
        if hasattr(mat, "matrix"):
            mat = mat.matrix
        mat_t = mat.T
        if sp.issparse(mat_t):
            mat_t = mat_t.tocsr()
        else:
            mat_t = np.asarray(mat_t)
        layers[aname] = mat_t

    # Choose primary X matrix: prefer logcounts / normcounts / data, else counts
    X_mat = None
    for preferred in ("logcounts", "normcounts", "data", "counts"):
        if preferred in layers:
            X_mat = layers[preferred]
            break
    if X_mat is None:
        X_mat = next(iter(layers.values()))

    n_cells, n_genes = X_mat.shape

    # Extract gene names (row_names or from raw assay Dimnames)
    gene_names = sce.get_row_names()
    if gene_names is not None:
        gene_names = [str(x) for x in gene_names]

    if not gene_names or len(gene_names) != n_genes:
        # Check raw assay Dimnames
        try:
            assay_list = attrs["assays"]["attributes"]["data"]["attributes"]["listData"]["data"]
            if assay_list:
                _, rnames, _ = _extract_matrix_and_dimnames(assay_list[0])
                if rnames and len(rnames) == n_genes:
                    gene_names = rnames
        except Exception:
            pass

    if not gene_names or len(gene_names) != n_genes:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

    # Extract cell names and column_data (obs)
    cell_names = sce.get_column_names()
    if cell_names is not None:
        cell_names = [str(x) for x in cell_names]
    if not cell_names or len(cell_names) != n_cells:
        cell_names = [f"Cell_{i}" for i in range(n_cells)]

    col_data = sce.get_column_data()
    obs_df = col_data.to_pandas() if hasattr(col_data, "to_pandas") else pd.DataFrame(index=cell_names)
    if "rownames" in obs_df.columns:
        obs_df = obs_df.drop(columns=["rownames"])
    obs_df.index = pd.Index(cell_names, name="cell_id")

    row_data = sce.get_row_data()
    var_df = row_data.to_pandas() if hasattr(row_data, "to_pandas") else pd.DataFrame(index=gene_names)
    if "rownames" in var_df.columns:
        var_df = var_df.drop(columns=["rownames"])
    var_df.index = pd.Index(gene_names, name="gene")

    adata = ad.AnnData(X=X_mat, obs=obs_df, var=var_df, layers=layers)

    # Extract reduced_dims
    if hasattr(sce, "get_reduced_dimension_names"):
        for rname in sce.get_reduced_dimension_names():
            rdim = sce.get_reduced_dimension(rname)
            if hasattr(rdim, "matrix"):
                rdim = rdim.matrix
            rdim_arr = np.asarray(rdim, dtype=np.float32)
            if rdim_arr.shape[0] == n_cells:
                key = f"X_{rname.lower()}" if not rname.lower().startswith("x_") else rname.lower()
                adata.obsm[key] = rdim_arr

    return adata


def _load_h5_or_h5ad(path: Path) -> ad.AnnData:
    """Load an .h5ad or 10x Genomics .h5 file into AnnData."""
    suffix = path.suffix.lower()
    if suffix == ".h5ad":
        return sc.read_h5ad(path)

    # Inspect HDF5 keys to distinguish AnnData .h5 from 10x CellRanger .h5
    with h5py.File(path, "r") as f:
        keys = set(f.keys())

    if "obs" in keys and ("X" in keys or "layers" in keys):
        return sc.read_h5ad(path)
    if "matrix" in keys:
        return sc.read_10x_h5(path)

    # Fallback: try read_10x_h5 first, then read_h5ad
    try:
        return sc.read_10x_h5(path)
    except Exception:
        return sc.read_h5ad(path)


def load_dataset(path: str | Path) -> ad.AnnData:
    """Load a single-cell dataset (.h5ad, .h5, or .rds) and prepare it for exploration."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in (".h5ad", ".h5"):
        adata = _load_h5_or_h5ad(file_path)
    elif suffix == ".rds":
        raw_rds = rds2py.parse_rds(str(file_path))
        class_name = raw_rds.get("class_name", "")
        if class_name == "Seurat":
            adata = _parse_seurat_rds(raw_rds)
        elif class_name in ("SingleCellExperiment", "SummarizedExperiment", "RangedSummarizedExperiment"):
            adata = _parse_sce_rds(raw_rds, file_path)
        else:
            raise ValueError(
                f"Unsupported RDS object class '{class_name}'. Expected 'Seurat' or 'SingleCellExperiment'."
            )
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Supported extensions: {SUPPORTED_EXTENSIONS}"
        )

    # Ensure unique gene and cell names
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # Convert categorical/object columns cleanly for plotting
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str).astype("category")
        elif isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].cat.remove_unused_categories()

    # If raw 10x counts were loaded (e.g. max value > 50 and integer-like), normalize & log1p
    # while saving raw counts in layers["counts"]
    if adata.X is not None and adata.n_obs > 0 and adata.n_vars > 0:
        sample_vals = (
            adata.X[: min(100, adata.n_obs), : min(100, adata.n_vars)].toarray()
            if sp.issparse(adata.X)
            else np.asarray(adata.X[: min(100, adata.n_obs), : min(100, adata.n_vars)])
        )
        max_val = float(np.nanmax(sample_vals)) if sample_vals.size > 0 else 0.0
        is_integer_like = np.allclose(sample_vals, np.round(sample_vals), atol=1e-3)
        if max_val > 30.0 and is_integer_like:
            if "counts" not in adata.layers:
                adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

    # Initialize default categorical color maps on first read:
    # Cell Annotations -> 'Paired', Samples -> 'Dark2', Conditions -> 'Accent'
    from scexplore.analysis import guess_metadata_columns
    from scexplore.plotting import build_dataset_color_maps

    guessed = guess_metadata_columns(adata)
    build_dataset_color_maps(
        adata,
        sample_col=guessed.get("sample"),
        condition_col=guessed.get("condition"),
        cell_type_col=guessed.get("cell_type"),
    )

    return adata
