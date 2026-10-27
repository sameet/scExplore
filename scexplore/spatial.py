"""Spatial transcriptomics, Scatter Pie, and SpatialCellChat Communication Field utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rds2py
from rds2py.generics import _dispatcher
import scipy.sparse as sp
from scipy.spatial import cKDTree

from scexplore.plotting import DISCRETE_PALETTE, build_category_color_map, resolve_discrete_color_map


# -----------------------------------------------------------------------------
# 1. Spatial Detection, Coordinate Extraction & Deconvolution Proportions
# -----------------------------------------------------------------------------

SPATIAL_COORD_PAIRS = [
    ("x_cent", "y_cent"),
    ("imagecol", "imagerow"),
    ("x", "y"),
    ("spatial_x", "spatial_y"),
    ("coord_x", "coord_y"),
    ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ("array_col", "array_row"),
]


def is_spatial_dataset(adata: ad.AnnData) -> bool:
    """Return True if the dataset has 2D spatial coordinates or SpatialCellChat data."""
    if get_spatial_coordinates(adata) is not None:
        return True
    if "spatial_cellchat" in adata.uns:
        return True
    return False


def get_spatial_coordinates(adata: ad.AnnData) -> pd.DataFrame | None:
    """Extract 2D spatial coordinates (`x`, `y`) aligned to `adata.obs_names`, if available."""
    for key in ("spatial", "X_spatial"):
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] == adata.n_obs and arr.shape[1] >= 2:
                return pd.DataFrame(
                    {"x": arr[:, 0], "y": arr[:, 1]},
                    index=adata.obs_names,
                )

    obs_lower = {c.lower(): c for c in adata.obs.columns}
    for cx, cy in SPATIAL_COORD_PAIRS:
        if cx in obs_lower and cy in obs_lower:
            col_x, col_y = obs_lower[cx], obs_lower[cy]
            if pd.api.types.is_numeric_dtype(adata.obs[col_x]) and pd.api.types.is_numeric_dtype(adata.obs[col_y]):
                arr = np.column_stack(
                    [
                        adata.obs[col_x].to_numpy(dtype=np.float64),
                        adata.obs[col_y].to_numpy(dtype=np.float64),
                    ]
                )
                adata.obsm["spatial"] = arr
                return pd.DataFrame({"x": arr[:, 0], "y": arr[:, 1]}, index=adata.obs_names)

    return None


def get_cell_type_proportions(
    adata: ad.AnnData,
    cell_type_col: str | None = None,
    grid_bins: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(proportions_df, coords_df)` for scatter pie plotting.

    - If precomputed deconvolution proportions exist in `adata.obsm['deconvolution']`,
      `adata.obsm['cell_type_proportions']`, `adata.uns['cell_type_proportions']`, or
      matching numeric columns in `adata.obs`, use those directly.
    - If `grid_bins > 0`, aggregate single cells into a `grid_bins x grid_bins` spatial grid
      and compute cell-type composition per grid tile (just like `makeGridSpatialCellChat`).
    - Otherwise, if a categorical `cell_type_col` is given, one-hot encode cell types per spot.
    """
    coords = get_spatial_coordinates(adata)
    if coords is None:
        raise ValueError("Dataset does not contain spatial coordinates.")

    # Check if grid aggregation is requested
    if grid_bins and grid_bins > 0 and cell_type_col and cell_type_col in adata.obs.columns:
        return aggregate_spatial_grid_proportions(adata, coords, cell_type_col, n_bins=grid_bins)

    # 1. Check obsm / uns for precomputed proportion tables
    for key in ("cell_type_proportions", "deconvolution", "proportions", "rctd", "cell2location", "spotlight"):
        if key in adata.obsm:
            val = adata.obsm[key]
            if isinstance(val, pd.DataFrame) and len(val) == adata.n_obs:
                return val.copy(), coords.copy()
            if isinstance(val, np.ndarray) and val.shape[0] == adata.n_obs:
                cols = adata.uns.get(f"{key}_columns", [f"Type_{i+1}" for i in range(val.shape[1])])
                return pd.DataFrame(val, index=adata.obs_names, columns=cols), coords.copy()
        if key in adata.uns and isinstance(adata.uns[key], pd.DataFrame):
            df_u = adata.uns[key]
            if len(df_u) == adata.n_obs:
                df_u = df_u.copy()
                df_u.index = adata.obs_names
                return df_u, coords.copy()

    # 2. Check if adata.obs contains columns matching the categories of cell_type_col (e.g. SpatialCellChat meta)
    if cell_type_col and cell_type_col in adata.obs.columns:
        ct_categories = [str(c) for c in adata.obs[cell_type_col].dropna().unique()]
        matching_num_cols = [
            c
            for c in ct_categories
            if c in adata.obs.columns and pd.api.types.is_numeric_dtype(adata.obs[c])
        ]
        if len(matching_num_cols) >= 2:
            prop_df = adata.obs[matching_num_cols].astype(np.float64).copy()
            row_sums = prop_df.sum(axis=1).replace(0, 1.0)
            prop_df = prop_df.div(row_sums, axis=0)
            return prop_df, coords.copy()

    # 3. Fallback: if dataset has many spots and user didn't set grid_bins, one-hot encode cell_type_col
    if cell_type_col and cell_type_col in adata.obs.columns:
        prop_df = pd.get_dummies(adata.obs[cell_type_col].astype(str), dtype=np.float64)
        prop_df.index = adata.obs_names
        return prop_df, coords.copy()

    raise ValueError("Could not find cell-type proportions or a valid cell type column for Scatter Pie plot.")


def aggregate_spatial_grid_proportions(
    adata: ad.AnnData,
    coords: pd.DataFrame,
    cell_type_col: str,
    n_bins: int = 15,
    region_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate individual spatial cells into a 2D spatial grid (like `makeGridSpatialCellChat`) and compute cell-type proportions per grid spot."""
    x = coords["x"].to_numpy()
    y = coords["y"].to_numpy()
    x_edges = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
    y_edges = np.linspace(y.min(), y.max() + 1e-9, n_bins + 1)

    bx = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    by = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)
    grid_ids = [f"Grid_{ix}_{iy}" for ix, iy in zip(bx, by)]

    df = pd.DataFrame(
        {
            "grid_id": grid_ids,
            "x": x,
            "y": y,
            "cell_type": adata.obs[cell_type_col].astype(str).to_numpy(),
        },
        index=adata.obs_names,
    )
    if region_col and region_col in adata.obs.columns:
        df["region"] = adata.obs[region_col].astype(str).to_numpy()

    counts = pd.crosstab(df["grid_id"], df["cell_type"])
    props = counts.div(counts.sum(axis=1).replace(0, 1.0), axis=0)

    centroids = df.groupby("grid_id")[["x", "y"]].mean()
    centroids["n_cells"] = counts.sum(axis=1)
    if "region" in df.columns:
        centroids["region"] = df.groupby("grid_id")["region"].agg(lambda s: s.mode().iloc[0] if len(s) > 0 else "All")

    centroids = centroids.loc[props.index]
    return props, centroids


# -----------------------------------------------------------------------------
# 2. Scatter Pie Plot Builder (with Region & Cell Type Selection)
# -----------------------------------------------------------------------------


def _make_wedge_polygon(
    cx: float,
    cy: float,
    radius: float,
    theta1: float,
    theta2: float,
    n_arc_pts: int = 14,
) -> tuple[list[float], list[float]]:
    """Return closed polygon `(xs, ys)` for a pie wedge from `theta1` to `theta2` (radians)."""
    if theta2 - theta1 >= 2 * np.pi - 1e-5:
        angles = np.linspace(0.0, 2 * np.pi, max(20, n_arc_pts * 2))
        xs = (cx + radius * np.cos(angles)).tolist() + [None]
        ys = (cy + radius * np.sin(angles)).tolist() + [None]
        return xs, ys

    angles = np.linspace(theta1, theta2, max(4, n_arc_pts))
    xs = [cx] + (cx + radius * np.cos(angles)).tolist() + [cx, None]
    ys = [cy] + (cy + radius * np.sin(angles)).tolist() + [cy, None]
    return xs, ys


def plot_scatter_pie(
    proportions_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    selected_cell_types: Sequence[str] | None = None,
    color_map: dict[str, str] | None = None,
    radius_scale: float = 0.85,
    renormalize_selected: bool = True,
    invert_y: bool = False,
    title: str = "Spatial Scatter Pie Plot",
) -> go.Figure:
    """Render an interactive spatial Scatter Pie plot (`visualizePie` equivalent) in Plotly.

    Supports filtering to specific regions (by passing subsetted `proportions_df` and `coords_df`)
    and selecting a subset of cell types (`selected_cell_types`).
    """
    available_types = [c for c in proportions_df.columns]
    use_types = (
        [c for c in selected_cell_types if c in available_types]
        if selected_cell_types
        else available_types
    )
    if not use_types:
        raise ValueError("Please select at least one cell type to display in the Scatter Pie plot.")

    sub_props = proportions_df[use_types].astype(np.float64).copy()
    row_totals = sub_props.sum(axis=1)
    valid_mask = row_totals > 1e-6
    sub_props = sub_props.loc[valid_mask]
    sub_coords = coords_df.loc[valid_mask]

    if sub_props.empty:
        fig = go.Figure()
        fig.update_layout(title="No spots with positive proportions in the selected region / cell types.")
        return fig

    if renormalize_selected:
        sub_props = sub_props.div(sub_props.sum(axis=1), axis=0)

    if not color_map:
        color_map = build_category_color_map(use_types, "Paired")

    x_vals = sub_coords["x"].to_numpy(dtype=np.float64)
    y_vals = sub_coords["y"].to_numpy(dtype=np.float64)
    if invert_y:
        y_vals = -y_vals

    # Compute default radius using SpatialCellChat::visualizePie formula
    n_spots = len(sub_coords)
    x_span = max(float(np.ptp(x_vals)), 1.0)
    y_span = max(float(np.ptp(y_vals)), 1.0)
    if n_spots > 1:
        tree = cKDTree(np.column_stack([x_vals, y_vals]))
        dists, _ = tree.query(np.column_stack([x_vals, y_vals]), k=2)
        nn_median = float(np.median(dists[:, 1]))
        base_radius = min(
            np.sqrt((x_span * y_span) / (n_spots * np.pi)) * 0.85,
            nn_median * 0.55 if nn_median > 0 else x_span * 0.03,
        )
    else:
        base_radius = max(x_span, y_span) * 0.05

    radius = max(base_radius * radius_scale, 1e-4)

    # Accumulate polygon coordinates per cell type so Plotly renders one fast trace per cell type
    poly_x: dict[str, list[float | None]] = {ct: [] for ct in use_types}
    poly_y: dict[str, list[float | None]] = {ct: [] for ct in use_types}

    prop_matrix = sub_props.to_numpy()
    for i in range(n_spots):
        cx, cy = float(x_vals[i]), float(y_vals[i])
        theta = 0.0
        for j, ct in enumerate(use_types):
            frac = float(prop_matrix[i, j])
            if frac <= 1e-4:
                continue
            dtheta = frac * 2.0 * np.pi
            n_pts = max(4, int(np.ceil(18 * frac)))
            wx, wy = _make_wedge_polygon(cx, cy, radius, theta, theta + dtheta, n_arc_pts=n_pts)
            poly_x[ct].extend(wx)
            poly_y[ct].extend(wy)
            theta += dtheta

    fig = go.Figure()
    for idx, ct in enumerate(use_types):
        color = color_map.get(ct, DISCRETE_PALETTE[idx % len(DISCRETE_PALETTE)])
        if not poly_x[ct]:
            continue
        fig.add_trace(
            go.Scatter(
                x=poly_x[ct],
                y=poly_y[ct],
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(color="rgba(255,255,255,0.35)", width=0.5),
                name=str(ct),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Add an invisible hover trace at spot centers showing exact cell-type composition percentages
    hover_texts: list[str] = []
    spot_names = sub_props.index.astype(str)
    for i in range(n_spots):
        parts = [f"<b>{spot_names[i]}</b>", f"x: {x_vals[i]:.1f}, y: {y_vals[i]:.1f}"]
        for j, ct in enumerate(use_types):
            pct = prop_matrix[i, j] * 100.0
            if pct >= 0.1:
                parts.append(f"{ct}: {pct:.1f}%")
        hover_texts.append("<br>".join(parts))

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(size=max(6.0, radius * 0.5), opacity=0.0, color="black"),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            name="Spot Composition",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=660,
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis=dict(title="Spatial X", showgrid=False, zeroline=False),
        yaxis=dict(
            title="Spatial Y",
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(title="Cell Type", itemsizing="constant"),
    )
    return fig


# -----------------------------------------------------------------------------
# 3. SpatialCellChat Analysis Object Parser & Communication Field Computation
# -----------------------------------------------------------------------------


def _named_list_to_dict(r_vec: dict[str, Any]) -> dict[str, Any]:
    """Convert an rds2py named vector/list into a `{name: element}` Python dict."""
    if not isinstance(r_vec, dict):
        return {}
    attrs = r_vec.get("attributes") or {}
    names_obj = attrs.get("names")
    data_list = r_vec.get("data")
    if names_obj and isinstance(names_obj, dict) and names_obj.get("data") is not None and isinstance(data_list, list):
        names = [str(x) for x in names_obj["data"]]
        return dict(zip(names, data_list))
    return {}


def _parse_sparse_3d_array(s3d_dict: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[str] | None]:
    """Parse an R `spatstat.sparse::sparse3Darray` from rds2py into `{pathway_name: (n_cells, 2) ndarray}`."""
    named = _named_list_to_dict(s3d_dict)
    if not named or "dims" not in named and "dim" not in named:
        return {}, None

    dim_key = "dim" if "dim" in named else "dims"
    dims = np.asarray(named[dim_key]["data"], dtype=int)
    if len(dims) != 3:
        return {}, None
    n_cells, n_coords, n_sig = int(dims[0]), int(dims[1]), int(dims[2])

    i_idx = np.asarray(named["i"]["data"], dtype=int) - 1
    j_idx = np.asarray(named["j"]["data"], dtype=int) - 1
    k_idx = np.asarray(named["k"]["data"], dtype=int) - 1
    x_val = np.asarray(named["x"]["data"], dtype=np.float64)

    dn_list = named.get("dimnames", {}).get("data", [])
    cell_names: list[str] | None = None
    sig_names: list[str] = [f"Pathway_{idx+1}" for idx in range(n_sig)]
    if len(dn_list) >= 1 and isinstance(dn_list[0], dict) and dn_list[0].get("data") is not None:
        cell_names = [str(c) for c in dn_list[0]["data"]]
    if len(dn_list) >= 3 and isinstance(dn_list[2], dict) and dn_list[2].get("data") is not None:
        sig_names = [str(s) for s in dn_list[2]["data"]]

    fields_by_sig: dict[str, np.ndarray] = {}
    for s_idx, s_name in enumerate(sig_names):
        arr = np.zeros((n_cells, max(2, n_coords)), dtype=np.float32)
        mask = k_idx == s_idx
        if np.any(mask):
            np.add.at(arr, (i_idx[mask], j_idx[mask]), x_val[mask])
        # Note: SpatialCellChat's netVisual_CommunField swaps columns 1 and 2 when both coordinates and field are swapped,
        # which keeps dx aligned with x_cent and dy aligned with y_cent.
        fields_by_sig[s_name] = arr[:, :2]

    return fields_by_sig, cell_names


def extract_spatial_cellchat_dict(raw_rds: dict[str, Any]) -> dict[str, Any]:
    """Extract coordinates, idents, meta, and communication fields (`netP` / `net`) from a parsed SpatialCellChat RDS dict."""
    attrs = raw_rds.get("attributes", {})
    result: dict[str, Any] = {
        "coordinates": None,
        "spatial_factors": {"ratio": 1.0, "tol": 5.0},
        "idents": None,
        "meta": None,
        "fields": {"netP": {"outgoing": {}, "incoming": {}}, "net": {"outgoing": {}, "incoming": {}}},
        "pathways": [],
    }

    # 1. Meta data
    if "meta" in attrs and attrs["meta"].get("type") != "null":
        try:
            meta_bf = _dispatcher(attrs["meta"])
            meta_df = meta_bf.to_pandas() if hasattr(meta_bf, "to_pandas") else pd.DataFrame(meta_bf)
            if "rownames" in meta_df.columns:
                meta_df.index = meta_df["rownames"].astype(str)
                meta_df = meta_df.drop(columns=["rownames"])
            result["meta"] = meta_df
        except Exception:
            pass

    # 2. Idents
    if "idents" in attrs and attrs["idents"].get("type") != "null":
        try:
            idents_val = _dispatcher(attrs["idents"])
            result["idents"] = [str(x) for x in idents_val]
        except Exception:
            pass

    # 3. Images -> coordinates and spatial.factors
    if "images" in attrs and attrs["images"].get("type") != "null":
        img_named = _named_list_to_dict(attrs["images"])
        if "coordinates" in img_named:
            try:
                coord_obj = _dispatcher(img_named["coordinates"])
                if hasattr(coord_obj, "to_pandas"):
                    cdf = coord_obj.to_pandas()
                    if "rownames" in cdf.columns:
                        cdf.index = cdf["rownames"].astype(str)
                        cdf = cdf.drop(columns=["rownames"])
                elif hasattr(coord_obj, "matrix"):
                    cdf = pd.DataFrame(coord_obj.matrix)
                else:
                    cdf = pd.DataFrame(np.asarray(coord_obj))
                if cdf.shape[1] >= 2:
                    cdf = cdf.iloc[:, :2].copy()
                    cdf.columns = ["x", "y"]
                    result["coordinates"] = cdf
            except Exception:
                pass

    # 4. Extract precomputed communication fields in netP and net
    for slot_name in ("netP", "net"):
        if slot_name not in attrs or attrs[slot_name].get("type") == "null":
            continue
        slot_named = _named_list_to_dict(attrs[slot_name])
        if "pathways" in slot_named and slot_named["pathways"].get("data") is not None:
            result["pathways"] = [str(p) for p in slot_named["pathways"]["data"]]

        if "field" in slot_named and slot_named["field"].get("type") != "null":
            field_named = _named_list_to_dict(slot_named["field"])
            for pattern in ("outgoing", "incoming"):
                if pattern in field_named:
                    f_dict, _ = _parse_sparse_3d_array(field_named[pattern])
                    result["fields"][slot_name][pattern].update(f_dict)

        # If `field` was not precomputed, check if `tmp$prob.cell` exists so we can compute it in Python!
        if (
            not result["fields"][slot_name]["outgoing"]
            and "tmp" in slot_named
            and slot_named["tmp"].get("type") != "null"
        ):
            tmp_named = _named_list_to_dict(slot_named["tmp"])
            if "prob.cell" in tmp_named and result["coordinates"] is not None:
                prob_cell_named = _named_list_to_dict(tmp_named["prob.cell"])
                coords_arr = result["coordinates"][["x", "y"]].to_numpy(dtype=np.float64)
                for sig_name, mat_dict in prob_cell_named.items():
                    try:
                        mat_disp = _dispatcher(mat_dict)
                        mat = mat_disp.matrix if hasattr(mat_disp, "matrix") else mat_disp
                        out_vec, in_vec = compute_communication_field_from_prob_matrix(mat, coords_arr, top=0.8)
                        result["fields"][slot_name]["outgoing"][sig_name] = out_vec
                        result["fields"][slot_name]["incoming"][sig_name] = in_vec
                    except Exception:
                        continue

    return result


def compute_communication_field_from_prob_matrix(
    prob_matrix: sp.spmatrix | np.ndarray,
    coords: np.ndarray,
    top: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Python implementation of `SpatialCellChat::computeCommunField` from a cell-by-cell communication probability matrix.

    Returns:
        `(outgoing_vectors, incoming_vectors)` each of shape `(n_cells, 2)`.
    """
    n_cells = coords.shape[0]
    outgoing = np.zeros((n_cells, 2), dtype=np.float32)
    incoming = np.zeros((n_cells, 2), dtype=np.float32)

    csr = sp.csr_matrix(prob_matrix) if not sp.isspmatrix_csr(prob_matrix) else prob_matrix
    csc = csr.tocsc()

    # Outgoing communication field per sender cell
    for i in range(n_cells):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        if start == end:
            continue
        targets = csr.indices[start:end]
        probs = csr.data[start:end]
        order = np.argsort(probs)[::-1]
        probs_sorted = probs[order]
        targets_sorted = targets[order]
        total_p = float(probs_sorted.sum())
        if total_p <= 0:
            continue
        csum = np.cumsum(probs_sorted)
        keep_mask = csum < top * total_p
        if not np.any(keep_mask):
            keep_mask[0] = True
        sel_targets = targets_sorted[keep_mask]
        sel_probs = probs_sorted[keep_mask]

        diffs = coords[sel_targets] - coords[i]
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        unit_dirs = np.where(norms > 1e-9, diffs / np.maximum(norms, 1e-9), 0.0)
        outgoing[i] = np.sum(unit_dirs * sel_probs[:, None], axis=0)

    # Incoming communication field per receiver cell
    for j in range(n_cells):
        start, end = csc.indptr[j], csc.indptr[j + 1]
        if start == end:
            continue
        senders = csc.indices[start:end]
        probs = csc.data[start:end]
        order = np.argsort(probs)[::-1]
        probs_sorted = probs[order]
        senders_sorted = senders[order]
        total_p = float(probs_sorted.sum())
        if total_p <= 0:
            continue
        csum = np.cumsum(probs_sorted)
        keep_mask = csum < top * total_p
        if not np.any(keep_mask):
            keep_mask[0] = True
        sel_senders = senders_sorted[keep_mask]
        sel_probs = probs_sorted[keep_mask]

        diffs = coords[j] - coords[sel_senders]
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        unit_dirs = np.where(norms > 1e-9, diffs / np.maximum(norms, 1e-9), 0.0)
        incoming[j] = np.sum(unit_dirs * sel_probs[:, None], axis=0)

    return outgoing, incoming


def compute_communication_field_from_lr_genes(
    adata: ad.AnnData,
    ligand_genes: Sequence[str],
    receptor_genes: Sequence[str],
    signaling_name: str = "Custom_LR",
    contact_knn: int = 12,
    length_scale: float | None = None,
    top: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a spatial cell-cell communication field directly on `adata` from Ligand and Receptor gene expression.

    Models short/medium-range spatial paracrine signaling `P(i -> j) = L_i * R_j * exp(-d_ij^2 / (2 * sigma^2))`
    across spatial k-nearest neighbors and runs `compute_communication_field_from_prob_matrix`.
    """
    coords_df = get_spatial_coordinates(adata)
    if coords_df is None:
        raise ValueError("Spatial coordinates are required to compute a spatial communication field.")
    coords = coords_df[["x", "y"]].to_numpy(dtype=np.float64)
    n_cells = coords.shape[0]

    def _mean_expr(genes: Sequence[str]) -> np.ndarray:
        valid = [g for g in genes if g in adata.var_names]
        if not valid:
            raise ValueError(f"None of the specified genes ({', '.join(genes)}) were found in the dataset.")
        sub = adata[:, valid].X
        arr = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        return np.clip(np.nanmean(arr, axis=1), 0.0, None).astype(np.float64)

    l_expr = _mean_expr(ligand_genes)
    r_expr = _mean_expr(receptor_genes)

    k_use = min(max(2, contact_knn + 1), n_cells)
    tree = cKDTree(coords)
    dists, nbrs = tree.query(coords, k=k_use)

    if length_scale is None or length_scale <= 0:
        positive_dists = dists[:, 1:]
        length_scale = float(np.median(positive_dists)) * 1.5 if positive_dists.size > 0 else 10.0
        length_scale = max(length_scale, 1e-3)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for i in range(n_cells):
        if l_expr[i] <= 1e-6:
            continue
        for m in range(1, k_use):
            j = int(nbrs[i, m])
            if r_expr[j] <= 1e-6:
                continue
            d = float(dists[i, m])
            w = float(l_expr[i] * r_expr[j] * np.exp(-(d * d) / (2.0 * length_scale * length_scale)))
            if w > 1e-6:
                rows.append(i)
                cols.append(j)
                vals.append(w)

    prob_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    out_vec, in_vec = compute_communication_field_from_prob_matrix(prob_mat, coords, top=top)

    # Store in adata.uns['spatial_cellchat']
    if "spatial_cellchat" not in adata.uns:
        adata.uns["spatial_cellchat"] = {
            "fields": {"netP": {"outgoing": {}, "incoming": {}}, "net": {"outgoing": {}, "incoming": {}}},
            "pathways": [],
        }
    scc = adata.uns["spatial_cellchat"]
    scc["fields"]["netP"]["outgoing"][signaling_name] = out_vec
    scc["fields"]["netP"]["incoming"][signaling_name] = in_vec
    existing_pathways = list(scc.get("pathways", []))
    if signaling_name not in existing_pathways:
        existing_pathways.append(signaling_name)
    scc["pathways"] = existing_pathways

    return out_vec, in_vec


def load_spatial_analysis_object(path: str | Path, target_adata: ad.AnnData | None = None) -> dict[str, Any]:
    """Load an external SpatialCellChat `.rds` or `.h5ad` analysis object and attach it to `target_adata` if provided."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Analysis file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".rds":
        raw_rds = rds2py.parse_rds(str(file_path))
        class_name = raw_rds.get("class_name", "")
        if class_name in ("SpatialCellChat", "CellChat"):
            scc_data = extract_spatial_cellchat_dict(raw_rds)
        else:
            raise ValueError(
                f"RDS file has class '{class_name}'. Expected 'SpatialCellChat' or 'CellChat'."
            )
    elif suffix in (".h5ad", ".h5"):
        import scanpy as sc

        aux_ad = sc.read_h5ad(file_path)
        scc_data = aux_ad.uns.get("spatial_cellchat", {})
        if not scc_data:
            # Check obsm for communication_field_* keys
            out_fields: dict[str, np.ndarray] = {}
            in_fields: dict[str, np.ndarray] = {}
            for k, v in aux_ad.obsm.items():
                if k.startswith("comm_field_out_"):
                    out_fields[k.replace("comm_field_out_", "")] = np.asarray(v)
                elif k.startswith("comm_field_in_"):
                    in_fields[k.replace("comm_field_in_", "")] = np.asarray(v)
            scc_data = {
                "coordinates": pd.DataFrame(aux_ad.obsm["spatial"], index=aux_ad.obs_names, columns=["x", "y"])
                if "spatial" in aux_ad.obsm
                else None,
                "idents": list(aux_ad.obs.iloc[:, 0].astype(str)) if aux_ad.obs.shape[1] > 0 else None,
                "meta": aux_ad.obs.copy(),
                "fields": {"netP": {"outgoing": out_fields, "incoming": in_fields}, "net": {"outgoing": {}, "incoming": {}}},
                "pathways": sorted(set(out_fields.keys()) | set(in_fields.keys())),
            }
    else:
        raise ValueError(f"Unsupported file extension '{suffix}' for SpatialCellChat object.")

    if target_adata is not None:
        target_adata.uns["spatial_cellchat"] = scc_data
        if scc_data.get("coordinates") is not None and len(scc_data["coordinates"]) == target_adata.n_obs:
            target_adata.obsm["spatial"] = scc_data["coordinates"][["x", "y"]].to_numpy(dtype=np.float64)

    return scc_data


def get_available_communication_signaling(adata: ad.AnnData) -> dict[str, list[str]]:
    """Return available communication field signaling names grouped by slot (`netP`, `net`)."""
    scc = adata.uns.get("spatial_cellchat", {})
    fields = scc.get("fields", {})
    out: dict[str, list[str]] = {}
    for slot_name in ("netP", "net"):
        slot_dict = fields.get(slot_name, {})
        names = sorted(
            set(slot_dict.get("outgoing", {}).keys()) | set(slot_dict.get("incoming", {}).keys())
        )
        if names:
            out[slot_name] = names
    return out


# -----------------------------------------------------------------------------
# 4. SpatialCellChat Communication Field & Grid Visualization (`netVisual_CommunField` & `netVisual_CommunFieldGrid`)
# -----------------------------------------------------------------------------


def _build_arrow_segments(
    x_starts: np.ndarray,
    y_starts: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    arrow_angle_deg: float = 22.5,
    head_fraction: float = 0.25,
) -> tuple[list[float | None], list[float | None]]:
    """Construct vector line + arrowhead segments `(xs, ys)` for a batch of 2D arrows."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    angle_rad = np.deg2rad(arrow_angle_deg)
    cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))

    for x0, y0, u, v in zip(x_starts, y_starts, dx, dy):
        mag = float(np.hypot(u, v))
        if mag <= 1e-9 or np.isnan(mag):
            continue
        x1 = float(x0 + u)
        y1 = float(y0 + v)

        # Shaft
        xs.extend([float(x0), x1, None])
        ys.extend([float(y0), y1, None])

        # Arrowhead wings
        hx = -u * head_fraction
        hy = -v * head_fraction
        wx1 = x1 + (hx * cos_a - hy * sin_a)
        wy1 = y1 + (hx * sin_a + hy * cos_a)
        wx2 = x1 + (hx * cos_a + hy * sin_a)
        wy2 = y1 + (-hx * sin_a + hy * cos_a)
        xs.extend([wx1, x1, wx2, None])
        ys.extend([wy1, y1, wy2, None])

    return xs, ys


def plot_communication_field(
    adata: ad.AnnData,
    signaling: str,
    slot_name: str = "netP",
    pattern: str = "outgoing",
    label_col: str | None = None,
    color_map: dict[str, str] | None = None,
    point_size: float = 5.0,
    point_alpha: float = 0.35,
    arrow_scale: float = 1.0,
    min_mag_percentile: float = 0.0,
    color_arrows_by_cell_type: bool = True,
    arrow_color: str = "#111111",
    invert_y: bool = True,
    region_mask: np.ndarray | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot cell-level SpatialCellChat Communication Field (`netVisual_CommunField` equivalent)."""
    scc = adata.uns.get("spatial_cellchat", {})
    field_dict = scc.get("fields", {}).get(slot_name, {}).get(pattern, {})
    if signaling not in field_dict:
        raise KeyError(
            f"Signaling '{signaling}' not found in `{slot_name}` ({pattern}) communication fields."
        )

    vec = np.asarray(field_dict[signaling], dtype=np.float64)
    coords_df = get_spatial_coordinates(adata)
    if coords_df is None and scc.get("coordinates") is not None:
        coords_df = scc["coordinates"]
    if coords_df is None:
        raise ValueError("No spatial coordinates found for Communication Field plot.")

    x = coords_df["x"].to_numpy(dtype=np.float64)
    y = coords_df["y"].to_numpy(dtype=np.float64)
    dx = vec[:, 0].copy()
    dy = vec[:, 1].copy()

    if invert_y:
        y = -y
        dy = -dy

    if label_col and label_col in adata.obs.columns:
        labels = adata.obs[label_col].astype(str).to_numpy()
    elif scc.get("idents") is not None and len(scc["idents"]) == len(x):
        labels = np.asarray(scc["idents"], dtype=str)
    else:
        labels = np.array(["Cells"] * len(x))

    if region_mask is not None and len(region_mask) == len(x):
        x = x[region_mask]
        y = y[region_mask]
        dx = dx[region_mask]
        dy = dy[region_mask]
        labels = labels[region_mask]

    mags = np.hypot(dx, dy)
    pos_mags = mags[mags > 1e-9]

    # Scale arrows cleanly to spatial coordinate extent
    span = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    if pos_mags.size > 0:
        p95 = float(np.percentile(pos_mags, 95))
        norm_factor = (span * 0.045 * arrow_scale) / max(p95, 1e-9)
        dx_scaled = dx * norm_factor
        dy_scaled = dy * norm_factor
    else:
        dx_scaled = dx
        dy_scaled = dy

    mag_threshold = float(np.percentile(pos_mags, min_mag_percentile)) if (pos_mags.size > 0 and min_mag_percentile > 0) else 1e-9

    if color_map is None and label_col:
        color_map = resolve_discrete_color_map(adata, label_col)
    unique_labels = sorted(set(labels.tolist()))
    if color_map is None:
        color_map = build_category_color_map(unique_labels, "Paired")

    fig = go.Figure()

    # 1. Background cell/spot scatter grouped by cell type
    for idx, cat in enumerate(unique_labels):
        c_mask = labels == cat
        color = color_map.get(cat, DISCRETE_PALETTE[idx % len(DISCRETE_PALETTE)])
        fig.add_trace(
            go.Scattergl(
                x=x[c_mask],
                y=y[c_mask],
                mode="markers",
                marker=dict(size=point_size, color=color, opacity=point_alpha),
                name=str(cat),
                legendgroup=str(cat),
                hovertemplate=(
                    f"<b>{cat}</b><br>x: %{{x:.1f}}, y: %{{y:.1f}}<br>"
                    "Field |v|: %{customdata:.4f}<extra></extra>"
                ),
                customdata=mags[c_mask],
            )
        )

    # 2. Communication field arrows
    active_mask = mags >= max(mag_threshold, 1e-9)
    if np.any(active_mask):
        if color_arrows_by_cell_type:
            for idx, cat in enumerate(unique_labels):
                cat_active = active_mask & (labels == cat)
                if not np.any(cat_active):
                    continue
                color = color_map.get(cat, DISCRETE_PALETTE[idx % len(DISCRETE_PALETTE)])
                axs, ays = _build_arrow_segments(
                    x[cat_active], y[cat_active], dx_scaled[cat_active], dy_scaled[cat_active]
                )
                fig.add_trace(
                    go.Scatter(
                        x=axs,
                        y=ays,
                        mode="lines",
                        line=dict(color=color, width=1.6),
                        name=f"{cat} arrows",
                        legendgroup=str(cat),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        else:
            axs, ays = _build_arrow_segments(
                x[active_mask], y[active_mask], dx_scaled[active_mask], dy_scaled[active_mask]
            )
            fig.add_trace(
                go.Scatter(
                    x=axs,
                    y=ays,
                    mode="lines",
                    line=dict(color=arrow_color, width=1.6),
                    name="Communication Vectors",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    legend_title = "Sources" if pattern == "outgoing" else "Targets"
    default_title = f"{pattern.capitalize()} Communication Field of {signaling}"
    fig.update_layout(
        title=title or default_title,
        template="plotly_white",
        height=660,
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis=dict(title="Spatial X", showgrid=False, zeroline=False),
        yaxis=dict(title="Spatial Y", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        legend=dict(title=legend_title, itemsizing="constant"),
    )
    return fig


def plot_communication_field_grid(
    adata: ad.AnnData,
    signaling: str,
    slot_name: str = "netP",
    pattern: str = "outgoing",
    grid_bins: int = 18,
    grid_agg: str = "median",
    smooth_field: bool = True,
    label_col: str | None = None,
    color_map: dict[str, str] | None = None,
    point_size: float = 5.0,
    point_alpha: float = 0.30,
    arrow_scale: float = 1.0,
    min_mag_percentile: float = 5.0,
    arrow_color: str = "#111111",
    invert_y: bool = True,
    region_mask: np.ndarray | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot gridded & smoothed SpatialCellChat Communication Flow (`netVisual_CommunFieldGrid` equivalent).

    Bins spatial coordinates into a 2D square grid, aggregates `(dx, dy)` per occupied tile using
    `median` or `sum`, optionally applies Barnes-style Gaussian spatial interpolation across occupied tiles,
    and overlays the vector field grid on top of the cell-type spatial scatter plot.
    """
    scc = adata.uns.get("spatial_cellchat", {})
    field_dict = scc.get("fields", {}).get(slot_name, {}).get(pattern, {})
    if signaling not in field_dict:
        raise KeyError(
            f"Signaling '{signaling}' not found in `{slot_name}` ({pattern}) communication fields."
        )

    vec = np.asarray(field_dict[signaling], dtype=np.float64)
    coords_df = get_spatial_coordinates(adata)
    if coords_df is None and scc.get("coordinates") is not None:
        coords_df = scc["coordinates"]
    if coords_df is None:
        raise ValueError("No spatial coordinates found for Communication Field Grid plot.")

    x = coords_df["x"].to_numpy(dtype=np.float64)
    y = coords_df["y"].to_numpy(dtype=np.float64)
    dx = vec[:, 0].copy()
    dy = vec[:, 1].copy()

    if label_col and label_col in adata.obs.columns:
        labels = adata.obs[label_col].astype(str).to_numpy()
    elif scc.get("idents") is not None and len(scc["idents"]) == len(x):
        labels = np.asarray(scc["idents"], dtype=str)
    else:
        labels = np.array(["Cells"] * len(x))

    if region_mask is not None and len(region_mask) == len(x):
        x = x[region_mask]
        y = y[region_mask]
        dx = dx[region_mask]
        dy = dy[region_mask]
        labels = labels[region_mask]

    if len(x) == 0:
        fig = go.Figure()
        fig.update_layout(title="No cells in selected region.")
        return fig

    # 1. Create 2D square grid over spatial extent
    n_bins = max(4, int(grid_bins))
    x_edges = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
    y_edges = np.linspace(y.min(), y.max() + 1e-9, n_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    bx = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    by = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)

    grid_df = pd.DataFrame({"bx": bx, "by": by, "dx": dx, "dy": dy})
    agg_fn = "median" if grid_agg == "median" else "sum"
    grouped = grid_df.groupby(["bx", "by"], as_index=False).agg(
        u=("dx", agg_fn),
        v=("dy", agg_fn),
        obs=("dx", "count"),
    )
    grouped = grouped[grouped["obs"] > 0].copy()
    grouped["gx"] = x_centers[grouped["bx"].to_numpy()]
    grouped["gy"] = y_centers[grouped["by"].to_numpy()]

    gx = grouped["gx"].to_numpy(dtype=np.float64)
    gy = grouped["gy"].to_numpy(dtype=np.float64)
    u = grouped["u"].to_numpy(dtype=np.float64)
    v = grouped["v"].to_numpy(dtype=np.float64)

    # Barnes-style spatial kernel smoothing over occupied grid centroids (like oce::interpBarnes + st_intersects mask)
    if smooth_field and len(gx) > 3:
        grid_pts = np.column_stack([gx, gy])
        step_x = (x.max() - x.min()) / n_bins
        step_y = (y.max() - y.min()) / n_bins
        sigma = max(step_x, step_y) * 1.15
        if sigma > 1e-9:
            diff_x = gx[:, None] - gx[None, :]
            diff_y = gy[:, None] - gy[None, :]
            dist2 = diff_x * diff_x + diff_y * diff_y
            weights = np.exp(-dist2 / (2.0 * sigma * sigma))
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
            u = weights @ u
            v = weights @ v

    plot_x = x.copy()
    plot_y = -y if invert_y else y.copy()
    plot_gy = -gy if invert_y else gy.copy()
    plot_v = -v if invert_y else v.copy()
    plot_u = u.copy()

    vel = np.hypot(plot_u, plot_v)
    pos_vel = vel[vel > 1e-9]

    # Scale grid arrows to match grid cell spacing
    cell_step = max((x.max() - x.min()) / n_bins, (y.max() - y.min()) / n_bins, 1e-3)
    if pos_vel.size > 0:
        p95 = float(np.percentile(pos_vel, 95))
        scale_factor = (cell_step * 0.95 * arrow_scale) / max(p95, 1e-9)
        u_scaled = plot_u * scale_factor
        v_scaled = plot_v * scale_factor
    else:
        u_scaled = plot_u
        v_scaled = plot_v

    mag_threshold = float(np.percentile(pos_vel, min_mag_percentile)) if (pos_vel.size > 0 and min_mag_percentile > 0) else 1e-9

    if color_map is None and label_col:
        color_map = resolve_discrete_color_map(adata, label_col)
    unique_labels = sorted(set(labels.tolist()))
    if color_map is None:
        color_map = build_category_color_map(unique_labels, "Paired")

    fig = go.Figure()

    # 1. Background cells colored by cell type
    for idx, cat in enumerate(unique_labels):
        c_mask = labels == cat
        color = color_map.get(cat, DISCRETE_PALETTE[idx % len(DISCRETE_PALETTE)])
        fig.add_trace(
            go.Scattergl(
                x=plot_x[c_mask],
                y=plot_y[c_mask],
                mode="markers",
                marker=dict(size=point_size, color=color, opacity=point_alpha),
                name=str(cat),
                legendgroup=str(cat),
                hovertemplate=f"<b>{cat}</b><br>x: %{{x:.1f}}, y: %{{y:.1f}}<extra></extra>",
            )
        )

    # 2. Gridded communication flow vectors
    active = vel >= max(mag_threshold, 1e-9)
    if np.any(active):
        axs, ays = _build_arrow_segments(
            gx[active], plot_gy[active], u_scaled[active], v_scaled[active], arrow_angle_deg=22.5, head_fraction=0.28
        )
        fig.add_trace(
            go.Scatter(
                x=axs,
                y=ays,
                mode="lines",
                line=dict(color=arrow_color, width=2.0),
                name="Grid Flow Vectors",
                hoverinfo="skip",
            )
        )
        # Add subtle hoverable centroids on grid tiles
        fig.add_trace(
            go.Scatter(
                x=gx[active],
                y=plot_gy[active],
                mode="markers",
                marker=dict(size=5, color=arrow_color, opacity=0.25),
                name="Grid Tile",
                showlegend=False,
                customdata=np.column_stack([vel[active], grouped["obs"].to_numpy()[active]]),
                hovertemplate=(
                    "<b>Grid Tile</b><br>x: %{x:.1f}, y: %{y:.1f}<br>"
                    "Flow Magnitude: %{customdata[0]:.4f}<br>"
                    "Cells in Tile: %{customdata[1]:.0f}<extra></extra>"
                ),
            )
        )

    legend_title = "Sources" if pattern == "outgoing" else "Targets"
    default_title = f"{pattern.capitalize()} Communication Flow Grid of {signaling}"
    fig.update_layout(
        title=title or default_title,
        template="plotly_white",
        height=660,
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis=dict(title="Spatial X", showgrid=True, gridcolor="rgba(200,200,200,0.25)", zeroline=False),
        yaxis=dict(
            title="Spatial Y",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.25)",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(title=legend_title, itemsizing="constant"),
    )
    return fig
