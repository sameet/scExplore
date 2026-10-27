"""Interactive Plotly visualization builders and palette utilities for scExplore."""

from __future__ import annotations

from typing import Sequence

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DISCRETE_PALETTE = (
    px.colors.qualitative.Bold
    + px.colors.qualitative.Vivid
    + px.colors.qualitative.Safe
    + px.colors.qualitative.Dark24
)

# Default continuous gradient (reverse RdGy) followed by user-selectable options
CONTINUOUS_GRADIENTS: list[str] = [
    "RdGy_r",
    "RdGy",
    "Viridis",
    "Plasma",
    "Magma",
    "Inferno",
    "Cividis",
    "RdBu_r",
    "YlOrRd",
    "Blues",
    "Reds",
]

DEFAULT_CONTINUOUS_GRADIENT = "RdGy_r"

ROLE_PALETTES: dict[str, str] = {
    "cell_type": "Paired",
    "sample": "Dark2",
    "condition": "Accent",
}


def get_palette_hex_list(palette_name: str, n_colors: int) -> list[str]:
    """Generate `n_colors` hex colors from a named Matplotlib qualitative colormap (e.g. 'Paired', 'Dark2', 'Accent')."""
    if n_colors <= 0:
        return []
    cmap = plt.get_cmap(palette_name)
    base_n = getattr(cmap, "N", 12)
    base_colors = [mcolors.to_hex(cmap(i)) for i in range(base_n)]
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    # Cycle through base colors if there are more categories than palette slots
    return [base_colors[i % len(base_colors)] for i in range(n_colors)]


def build_category_color_map(
    categories: Sequence[str],
    palette_name: str,
) -> dict[str, str]:
    """Build a deterministic `{category_str: hex_color}` mapping using `palette_name`."""
    unique_cats = sorted({str(c) for c in categories if pd.notna(c)})
    colors = get_palette_hex_list(palette_name, len(unique_cats))
    return dict(zip(unique_cats, colors))


def build_dataset_color_maps(
    adata: ad.AnnData,
    sample_col: str | None = None,
    condition_col: str | None = None,
    cell_type_col: str | None = None,
) -> dict[str, dict[str, str]]:
    """Create and store persistent categorical color maps for cell_type ('Paired'), sample ('Dark2'), and condition ('Accent').

    Also stores the color maps in `adata.uns['scexplore_color_maps']`.
    """
    color_maps: dict[str, dict[str, str]] = {}

    role_to_col = {
        "cell_type": cell_type_col,
        "sample": sample_col,
        "condition": condition_col,
    }

    for role, col_name in role_to_col.items():
        if col_name and col_name in adata.obs.columns:
            palette_name = ROLE_PALETTES[role]
            cmap = build_category_color_map(adata.obs[col_name].astype(str).unique(), palette_name)
            color_maps[col_name] = cmap
            # Also store by role key so lookup works by either column name or role
            color_maps[f"__role_{role}__"] = cmap

    adata.uns["scexplore_color_maps"] = color_maps
    return color_maps


def resolve_discrete_color_map(
    adata: ad.AnnData,
    col_name: str | None,
    color_maps: dict[str, dict[str, str]] | None = None,
) -> dict[str, str] | None:
    """Resolve the discrete color map for a metadata column from `color_maps` or `adata.uns`."""
    if not col_name:
        return None
    maps = color_maps or adata.uns.get("scexplore_color_maps", {})
    if col_name in maps:
        return maps[col_name]
    return None


def plot_embedding(
    adata: ad.AnnData,
    obsm_key: str,
    color_by: str | pd.Series | None = None,
    title: str | None = None,
    color_scale: str = DEFAULT_CONTINUOUS_GRADIENT,
    color_discrete_map: dict[str, str] | None = None,
    point_size: float = 5.0,
    opacity: float = 0.85,
    sort_order: bool = True,
) -> go.Figure:
    """Plot a 2D dimensionality reduction (UMAP, PHATE, etc.) colored by metadata, gene expression, or module score."""
    coords = np.asarray(adata.obsm[obsm_key])
    dim_prefix = obsm_key[2:].upper() if obsm_key.lower().startswith("x_") else obsm_key.upper()
    col_x = f"{dim_prefix}_1"
    col_y = f"{dim_prefix}_2"

    df = pd.DataFrame(
        {
            col_x: coords[:, 0],
            col_y: coords[:, 1],
            "cell_id": adata.obs_names.astype(str),
        },
        index=adata.obs_names,
    )

    color_col = None
    is_continuous = False

    if isinstance(color_by, pd.Series):
        color_col = color_by.name or "Value"
        df[color_col] = color_by.reindex(adata.obs_names).values
        is_continuous = pd.api.types.is_numeric_dtype(df[color_col])
    elif isinstance(color_by, str) and color_by in adata.obs.columns:
        color_col = color_by
        s = adata.obs[color_by]
        if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype) and s.nunique() > 25:
            df[color_col] = s.values
            is_continuous = True
        else:
            df[color_col] = s.astype(str).values
            is_continuous = False

    if is_continuous and color_col and sort_order:
        # Sort so highest-expressing cells are plotted on top
        df = df.sort_values(by=color_col, ascending=True, na_position="first")

    plot_title = title or (f"{dim_prefix} — {color_col}" if color_col else dim_prefix)

    if color_col is None:
        fig = px.scatter(
            df,
            x=col_x,
            y=col_y,
            hover_data=["cell_id"],
            title=plot_title,
            render_mode="webgl",
        )
    elif is_continuous:
        fig = px.scatter(
            df,
            x=col_x,
            y=col_y,
            color=color_col,
            color_continuous_scale=color_scale,
            hover_data=["cell_id"],
            title=plot_title,
            render_mode="webgl",
        )
    else:
        resolved_cmap = color_discrete_map or resolve_discrete_color_map(adata, color_col)
        categories = sorted(df[color_col].dropna().unique().tolist())
        fig = px.scatter(
            df,
            x=col_x,
            y=col_y,
            color=color_col,
            category_orders={color_col: categories},
            color_discrete_map=resolved_cmap,
            color_discrete_sequence=DISCRETE_PALETTE if not resolved_cmap else None,
            hover_data=["cell_id"],
            title=plot_title,
            render_mode="webgl",
        )

    fig.update_traces(
        marker=dict(size=point_size, opacity=opacity, line=dict(width=0)),
    )
    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        legend=dict(itemsizing="constant"),
    )
    return fig


def plot_gene_violin(
    adata: ad.AnnData,
    expr_series: pd.Series,
    x_col: str,
    color_col: str | None = None,
    color_discrete_map: dict[str, str] | None = None,
    box: bool = True,
    points: str | bool = "outliers",
    title: str | None = None,
) -> go.Figure:
    """Create a violin plot of gene expression grouped by `x_col` and optionally colored by `color_col`."""
    gene_name = expr_series.name or "Expression"
    df = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            gene_name: expr_series.reindex(adata.obs_names).values,
            x_col: adata.obs[x_col].astype(str).values,
        },
        index=adata.obs_names,
    )

    effective_color = color_col if (color_col and color_col in adata.obs.columns) else x_col
    if effective_color != x_col:
        df[effective_color] = adata.obs[effective_color].astype(str).values

    # Sort categories alphabetically/naturally for clean x-axis ordering
    x_categories = sorted(df[x_col].dropna().unique().tolist())
    category_orders = {x_col: x_categories}
    if effective_color != x_col:
        category_orders[effective_color] = sorted(df[effective_color].dropna().unique().tolist())

    plot_title = title or (
        f"{gene_name} Expression by {x_col}"
        + (f" (colored by {effective_color})" if effective_color != x_col else "")
    )

    violin_mode = "overlay" if effective_color == x_col else "group"
    resolved_cmap = color_discrete_map or resolve_discrete_color_map(adata, effective_color)

    fig = px.violin(
        df,
        x=x_col,
        y=gene_name,
        color=effective_color,
        box=box,
        points=points,
        category_orders=category_orders,
        color_discrete_map=resolved_cmap,
        color_discrete_sequence=DISCRETE_PALETTE if not resolved_cmap else None,
        hover_data=["cell_id"],
        title=plot_title,
    )
    fig.update_layout(
        violinmode=violin_mode,
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=55, b=80),
        xaxis_tickangle=-35,
        yaxis_title=f"{gene_name} Expression",
        xaxis_title=x_col,
    )
    return fig
