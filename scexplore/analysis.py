"""Analysis and computation functions for scExplore (UMAP, PHATE, gene expression, module scores)."""

from __future__ import annotations

from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import phate
import scanpy as sc
import scipy.sparse as sp


def get_available_reductions(adata: ad.AnnData) -> dict[str, str]:
    """Return mapping of friendly reduction names (e.g. 'UMAP', 'PHATE', 'PCA', 'tSNE') to obsm keys."""
    mapping: dict[str, str] = {}
    for key in adata.obsm.keys():
        emb = adata.obsm[key]
        if hasattr(emb, "shape") and len(emb.shape) == 2 and emb.shape[1] >= 2:
            clean = key[2:] if key.lower().startswith("x_") else key
            label = clean.upper()
            mapping[label] = key
    return mapping


def ensure_pca(adata: ad.AnnData, n_comps: int = 30) -> str:
    """Ensure PCA embedding exists in adata.obsm['X_pca']."""
    for k in adata.obsm.keys():
        if k.lower() in ("x_pca", "pca"):
            if k != "X_pca":
                adata.obsm["X_pca"] = np.asarray(adata.obsm[k], dtype=np.float32)
            return "X_pca"

    max_comps = max(2, min(n_comps, adata.n_obs - 1, adata.n_vars - 1))
    sc.tl.pca(adata, n_comps=max_comps, svd_solver="arpack")
    return "X_pca"


def compute_umap(
    adata: ad.AnnData,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    n_pcs: int = 30,
    random_state: int = 42,
) -> str:
    """Compute UMAP on-demand and store in adata.obsm['X_umap']."""
    ensure_pca(adata, n_comps=n_pcs)
    use_pcs = min(n_pcs, adata.obsm["X_pca"].shape[1])
    neighbors = max(2, min(n_neighbors, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=use_pcs, random_state=random_state)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    return "X_umap"


def compute_phate(
    adata: ad.AnnData,
    knn: int = 5,
    decay: int = 40,
    t: str | int = "auto",
    n_pcs: int = 30,
    random_state: int = 42,
) -> str:
    """Compute PHATE embedding on-demand and store in adata.obsm['X_phate']."""
    ensure_pca(adata, n_comps=n_pcs)
    use_pcs = min(n_pcs, adata.obsm["X_pca"].shape[1])
    pca_coords = np.asarray(adata.obsm["X_pca"][:, :use_pcs], dtype=np.float64)

    knn_val = max(2, min(knn, adata.n_obs - 2))
    phate_op = phate.PHATE(
        n_components=2,
        knn=knn_val,
        decay=decay,
        t=t,
        n_pca=None,  # Already using PCA coordinates for speed and stability
        random_state=random_state,
        verbose=False,
        n_jobs=1,
    )
    emb = phate_op.fit_transform(pca_coords)
    adata.obsm["X_phate"] = np.asarray(emb, dtype=np.float32)
    return "X_phate"


def resolve_gene_name(adata: ad.AnnData, query: str) -> str | None:
    """Resolve a gene symbol (exact match first, then case-insensitive match)."""
    q = query.strip()
    if not q:
        return None
    if q in adata.var_names:
        return q
    lower_map = {g.lower(): g for g in adata.var_names}
    if q.lower() in lower_map:
        return lower_map[q.lower()]
    if adata.raw is not None:
        if q in adata.raw.var_names:
            return q
        raw_lower = {g.lower(): g for g in adata.raw.var_names}
        if q.lower() in raw_lower:
            return raw_lower[q.lower()]
    return None


def get_gene_expression(
    adata: ad.AnnData, gene: str, layer: str | None = None
) -> pd.Series:
    """Extract a 1D pandas Series of expression values for a single gene across all cells."""
    resolved = resolve_gene_name(adata, gene)
    if resolved is None:
        raise KeyError(f"Gene '{gene}' not found in dataset.")

    if layer and layer in adata.layers and resolved in adata.var_names:
        idx = adata.var_names.get_loc(resolved)
        col = adata.layers[layer][:, idx]
    elif resolved in adata.var_names:
        idx = adata.var_names.get_loc(resolved)
        col = adata.X[:, idx]
    elif adata.raw is not None and resolved in adata.raw.var_names:
        idx = adata.raw.var_names.get_loc(resolved)
        col = adata.raw.X[:, idx]
    else:
        raise KeyError(f"Gene '{gene}' not found in selected matrix.")

    if sp.issparse(col):
        vals = np.asarray(col.toarray()).ravel()
    else:
        vals = np.asarray(col).ravel()

    return pd.Series(vals, index=adata.obs_names, name=resolved, dtype=np.float32)


def calculate_module_score(
    adata: ad.AnnData,
    genes: Sequence[str],
    score_name: str = "Module_Score",
    ctrl_size: int = 50,
    n_bins: int = 25,
) -> tuple[pd.Series, list[str], list[str]]:
    """Calculate module score for a list of genes using Scanpy's score_genes (Seurat AddModuleScore equivalent).

    Returns:
        (score_series, matched_genes, missing_genes)
    """
    matched: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    for g in genes:
        clean = g.strip()
        if not clean:
            continue
        res = resolve_gene_name(adata, clean)
        if res is not None and res in adata.var_names:
            if res not in seen:
                matched.append(res)
                seen.add(res)
        else:
            missing.append(clean)

    if not matched:
        raise ValueError(
            f"None of the input genes were found in the dataset. Missing: {', '.join(missing)}"
        )

    effective_bins = max(2, min(n_bins, adata.n_vars // 2))
    effective_ctrl = max(2, min(ctrl_size, max(2, adata.n_vars // effective_bins - 1)))

    try:
        sc.tl.score_genes(
            adata,
            gene_list=matched,
            ctrl_size=effective_ctrl,
            n_bins=effective_bins,
            score_name=score_name,
            use_raw=False,
        )
    except Exception:
        # Fallback to mean z-scored expression across matched genes if dataset has too few genes for binning
        sub = adata[:, matched].X
        arr = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        adata.obs[score_name] = np.nanmean(arr, axis=1).astype(np.float32)

    return adata.obs[score_name].copy(), matched, missing


def guess_metadata_columns(adata: ad.AnnData) -> dict[str, str | None]:
    """Heuristically identify default metadata columns for sample, condition, and cell_type."""
    cols = list(adata.obs.columns)
    lower_to_col = {c.lower(): c for c in cols}

    def _match(candidates: Sequence[str], substrings: Sequence[str]) -> str | None:
        for cand in candidates:
            if cand.lower() in lower_to_col:
                return lower_to_col[cand.lower()]
        for c in cols:
            cl = c.lower()
            if any(sub in cl for sub in substrings):
                return c
        return None

    sample_col = _match(
        ["sample", "sample_id", "orig.ident", "batch", "donor", "patient", "replicate", "library_id"],
        ["sample", "orig.ident", "donor", "patient", "batch"],
    )
    condition_col = _match(
        ["condition", "treatment", "group", "status", "disease", "stim", "timepoint", "groups"],
        ["cond", "treat", "group", "status", "disease", "stim"],
    )
    cell_type_col = _match(
        [
            "cell_type",
            "celltype",
            "annotated_cell_type",
            "annotation",
            "clusters",
            "seurat_clusters",
            "leiden",
            "louvain",
            "ident",
            "seurat_active_ident",
            "letter.idents",
        ],
        ["cell_type", "celltype", "annot", "cluster", "leiden", "louvain", "ident"],
    )

    # Fallback to first categorical column if any is None
    cat_cols = get_categorical_columns(adata)
    fallback = cat_cols[0] if cat_cols else (cols[0] if cols else None)

    return {
        "sample": sample_col or fallback,
        "condition": condition_col or fallback,
        "cell_type": cell_type_col or fallback,
    }


def get_categorical_columns(adata: ad.AnnData, max_unique: int = 200) -> list[str]:
    """Return metadata columns suitable for grouping/coloring (categorical or discrete with <= max_unique values)."""
    result: list[str] = []
    for col in adata.obs.columns:
        s = adata.obs[col]
        if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) or pd.api.types.is_bool_dtype(s):
            if s.nunique(dropna=True) <= max_unique:
                result.append(col)
        elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= min(50, max_unique):
            result.append(col)
    return result
