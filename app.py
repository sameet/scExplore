"""Main Streamlit browser application for scExplore."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

from scexplore.analysis import (
    calculate_module_score,
    compute_phate,
    compute_umap,
    get_available_reductions,
    get_categorical_columns,
    get_gene_expression,
    guess_metadata_columns,
    resolve_gene_name,
)
from scexplore.data_loader import SUPPORTED_EXTENSIONS, list_sc_files, load_dataset
from scexplore.directory_picker import generate_demo_dataset, pick_directory_dialog
from scexplore.plotting import (
    CONTINUOUS_GRADIENTS,
    DEFAULT_CONTINUOUS_GRADIENT,
    build_category_color_map,
    build_dataset_color_maps,
    plot_embedding,
    plot_gene_violin,
    resolve_discrete_color_map,
)
from scexplore.spatial import (
    compute_communication_field_from_lr_genes,
    get_available_communication_signaling,
    get_cell_type_proportions,
    get_spatial_coordinates,
    is_spatial_dataset,
    load_spatial_analysis_object,
    plot_communication_field,
    plot_communication_field_grid,
    plot_scatter_pie,
)


st.set_page_config(
    page_title="scExplore — Single-Cell RNA-Seq Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session_state() -> None:
    if "current_dir" not in st.session_state:
        default_sample_dir = Path.cwd() / "sample_data"
        st.session_state["current_dir"] = str(
            default_sample_dir if default_sample_dir.exists() else Path.cwd()
        )
    if "loaded_file" not in st.session_state:
        st.session_state["loaded_file"] = None
    if "adata" not in st.session_state:
        st.session_state["adata"] = None
    if "module_scores" not in st.session_state:
        st.session_state["module_scores"] = {}
    if "color_maps" not in st.session_state:
        st.session_state["color_maps"] = {}


def _load_file_into_state(file_path: Path) -> None:
    with st.spinner(f"Loading `{file_path.name}`..."):
        adata = load_dataset(file_path)
        st.session_state["adata"] = adata
        st.session_state["loaded_file"] = str(file_path)
        st.session_state["module_scores"] = {}
        st.session_state["color_maps"] = dict(adata.uns.get("scexplore_color_maps", {}))


def render_sidebar() -> None:
    """Render sidebar for directory selection, dataset loading, and metadata column mapping."""
    st.sidebar.title("🧬 scExplore")
    st.sidebar.caption("Interactive Single-Cell RNA-Seq Browser")

    st.sidebar.markdown("### 1. Select Data Directory")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("📁 Browse Folder", use_container_width=True, help="Open native OS folder picker"):
            chosen = pick_directory_dialog(st.session_state["current_dir"])
            if chosen:
                st.session_state["current_dir"] = str(Path(chosen).expanduser().resolve())
                st.rerun()
    with col_btn2:
        if st.button("🧪 Generate Demo", use_container_width=True, help="Create synthetic merged PBMC dataset"):
            sample_dir = Path.cwd() / "sample_data"
            demo_path = generate_demo_dataset(sample_dir)
            st.session_state["current_dir"] = str(sample_dir)
            _load_file_into_state(demo_path)
            st.rerun()

    # Quick navigation & manual path input
    cur_path = Path(st.session_state["current_dir"]).expanduser()
    dir_input = st.sidebar.text_input(
        "Directory Path",
        value=str(cur_path),
        help="Enter a directory path or use Browse Folder / subfolder navigation below.",
    )
    if dir_input != str(cur_path):
        cand = Path(dir_input).expanduser()
        if cand.exists() and cand.is_dir():
            st.session_state["current_dir"] = str(cand.resolve())
            st.rerun()
        else:
            st.sidebar.warning("Directory does not exist.")

    # Quick subfolder / parent navigation expander
    with st.sidebar.expander("📂 Navigate Subfolders", expanded=False):
        if st.button("⬆️ Parent Directory", use_container_width=True):
            st.session_state["current_dir"] = str(cur_path.parent.resolve())
            st.rerun()
        if cur_path.exists() and cur_path.is_dir():
            subdirs = sorted(
                [d for d in cur_path.iterdir() if d.is_dir() and not d.name.startswith(".")],
                key=lambda d: d.name.lower(),
            )
            if subdirs:
                chosen_sub = st.selectbox(
                    "Jump into subfolder",
                    options=["(stay in current)"] + [d.name for d in subdirs],
                    index=0,
                )
                if chosen_sub != "(stay in current)":
                    st.session_state["current_dir"] = str((cur_path / chosen_sub).resolve())
                    st.rerun()
            else:
                st.caption("No subdirectories found.")

    recursive_scan = st.sidebar.checkbox("Scan subdirectories recursively", value=False)
    sc_files = list_sc_files(st.session_state["current_dir"], recursive=recursive_scan)

    st.sidebar.markdown("### 2. Load Experiment File")
    if not sc_files:
        st.sidebar.info(
            f"No `{', '.join(SUPPORTED_EXTENSIONS)}` files found in `{cur_path.name}`.\n\n"
            "Click **🧪 Generate Demo** above to create a sample multi-condition PBMC experiment, "
            "or browse to a directory with `.rds`, `.h5`, or `.h5ad` files."
        )
        return

    base_dir = Path(st.session_state["current_dir"]).resolve()
    file_labels = [str(f.relative_to(base_dir)) if f.is_relative_to(base_dir) else f.name for f in sc_files]
    label_to_path = dict(zip(file_labels, sc_files))

    default_idx = 0
    if st.session_state["loaded_file"]:
        for i, f in enumerate(sc_files):
            if str(f) == st.session_state["loaded_file"]:
                default_idx = i
                break

    selected_label = st.sidebar.selectbox(
        "Available Data Files (.rds, .h5, .h5ad)",
        options=file_labels,
        index=default_idx,
    )
    selected_path = label_to_path[selected_label]

    load_clicked = st.sidebar.button("🚀 Load Dataset", type="primary", use_container_width=True)
    if load_clicked or (st.session_state["adata"] is None and sc_files):
        try:
            _load_file_into_state(selected_path)
        except Exception as exc:
            st.sidebar.error(f"Failed to load `{selected_path.name}`: {exc}")
            return

    adata = st.session_state["adata"]
    if adata is None:
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 3. Metadata Mapping")
    cat_cols = get_categorical_columns(adata)
    all_obs_cols = list(adata.obs.columns)
    selectable_cols = cat_cols if cat_cols else all_obs_cols

    if selectable_cols:
        guessed = guess_metadata_columns(adata)

        def _idx(col_name: str | None) -> int:
            if col_name and col_name in selectable_cols:
                return selectable_cols.index(col_name)
            return 0

        st.session_state["meta_sample"] = st.sidebar.selectbox(
            "Sample Column (Dark2 Palette)",
            options=selectable_cols,
            index=_idx(guessed["sample"]),
            help="Metadata column identifying individual samples / replicates (mapped to 'Dark2' palette)",
        )
        st.session_state["meta_condition"] = st.sidebar.selectbox(
            "Condition Column (Accent Palette)",
            options=selectable_cols,
            index=_idx(guessed["condition"]),
            help="Metadata column identifying experimental condition / group / treatment (mapped to 'Accent' palette)",
        )
        st.session_state["meta_cell_type"] = st.sidebar.selectbox(
            "Annotated Cell Type Column (Paired Palette)",
            options=selectable_cols,
            index=_idx(guessed["cell_type"]),
            help="Metadata column identifying annotated cell types or clusters (mapped to 'Paired' palette)",
        )

        # Ensure color maps reflect the active column assignments:
        # Cell Annotations -> 'Paired', Samples -> 'Dark2', Conditions -> 'Accent'
        st.session_state["color_maps"] = build_dataset_color_maps(
            adata,
            sample_col=st.session_state["meta_sample"],
            condition_col=st.session_state["meta_condition"],
            cell_type_col=st.session_state["meta_cell_type"],
        )
    else:
        st.sidebar.warning("No cell metadata columns found in dataset.")


def render_reduction_controls(adata, key_prefix: str) -> tuple[str | None, float, str]:
    """Render dimensionality reduction selector (UMAP / PHATE / others) with on-demand compute."""
    avail = get_available_reductions(adata)

    # Always include UMAP and PHATE as selectable targets
    reduction_choices = ["UMAP", "PHATE"]
    for k in avail.keys():
        if k not in reduction_choices:
            reduction_choices.append(k)

    c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 2])
    with c1:
        selected_red = st.selectbox(
            "Dimensionality Reduction",
            options=reduction_choices,
            index=0,
            key=f"{key_prefix}_red_select",
        )
    with c2:
        point_size = st.slider(
            "Point Size",
            min_value=1.0,
            max_value=12.0,
            value=5.0,
            step=0.5,
            key=f"{key_prefix}_pt_size",
        )
    with c3:
        color_scale = st.selectbox(
            "Gradient Palette",
            options=CONTINUOUS_GRADIENTS,
            index=CONTINUOUS_GRADIENTS.index(DEFAULT_CONTINUOUS_GRADIENT),
            format_func=lambda g: f"{g} (Reverse RdGy)" if g == "RdGy_r" else g,
            key=f"{key_prefix}_cscale",
        )

    obsm_key = avail.get(selected_red)
    with c4:
        st.write("")
        btn_label = (
            f"⚡ Recompute {selected_red}"
            if obsm_key is not None
            else f"⚡ Compute {selected_red} Now"
        )
        if selected_red in ("UMAP", "PHATE"):
            if st.button(btn_label, key=f"{key_prefix}_compute_btn", use_container_width=True):
                with st.spinner(f"Computing {selected_red} embedding..."):
                    if selected_red == "UMAP":
                        obsm_key = compute_umap(adata)
                    else:
                        obsm_key = compute_phate(adata)
                st.rerun()

    if obsm_key is None:
        st.info(
            f"**{selected_red}** coordinates are not precomputed in this dataset. "
            f"Click **⚡ Compute {selected_red} Now** above to compute it on-demand."
        )

    return obsm_key, point_size, color_scale


def main() -> None:
    _init_session_state()
    render_sidebar()

    adata = st.session_state.get("adata")
    loaded_file = st.session_state.get("loaded_file")

    st.title("🧬 scExplore: Single-Cell RNA-Seq Analysis & Visualization")

    if adata is None:
        st.info(
            "👈 Select a folder and load an `.rds` (Seurat / SingleCellExperiment), `.h5` (10x Genomics / AnnData), "
            "or `.h5ad` file from the sidebar, or click **🧪 Generate Demo** to explore a multi-sample PBMC dataset."
        )
        return

    # Dataset Summary Banner
    avail_reds = get_available_reductions(adata)
    has_spatial = is_spatial_dataset(adata)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Loaded Experiment", Path(loaded_file).name if loaded_file else "Dataset")
    m2.metric("Cells / Spots", f"{adata.n_obs:,}")
    m3.metric("Genes (Features)", f"{adata.n_vars:,}")
    m4.metric(
        "Embeddings / Spatial",
        (", ".join(avail_reds.keys()) if avail_reds else "None") + (" + Spatial" if has_spatial else ""),
    )

    tab_embed, tab_gene, tab_module, tab_violin, tab_spatial = st.tabs(
        [
            "🗺️ UMAP & PHATE Explorer",
            "🔍 Single-Gene Expression",
            "🧮 Multi-Gene Module Score",
            "🎻 Violin Plots (Sample, Condition & Cell Type)",
            "🧭 Spatial & CellChat Explorer",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: UMAP & PHATE Explorer
    # -------------------------------------------------------------------------
    with tab_embed:
        st.subheader("Dimensionality Reduction Explorer (UMAP & PHATE)")
        st.caption(
            "Display precomputed embeddings or compute UMAP and PHATE on-demand, colored by any cell metadata column."
        )

        col_ctrl1, col_ctrl2 = st.columns([2, 2])
        all_obs_cols = list(adata.obs.columns)
        default_color = st.session_state.get("meta_cell_type") or (all_obs_cols[0] if all_obs_cols else None)
        with col_ctrl1:
            color_meta = st.selectbox(
                "Color Cells By Metadata",
                options=all_obs_cols if all_obs_cols else ["None"],
                index=all_obs_cols.index(default_color) if default_color in all_obs_cols else 0,
                key="tab1_color_meta",
            )
        with col_ctrl2:
            view_mode = st.radio(
                "Display Layout",
                options=["Single Embedding View", "Side-by-Side (UMAP & PHATE)"],
                horizontal=True,
                key="tab1_view_mode",
            )

        if view_mode == "Single Embedding View":
            obsm_key, pt_size, cscale = render_reduction_controls(adata, key_prefix="tab1")
            if obsm_key is not None:
                fig = plot_embedding(
                    adata,
                    obsm_key=obsm_key,
                    color_by=color_meta if color_meta in adata.obs.columns else None,
                    point_size=pt_size,
                    color_scale=cscale,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Side-by-side UMAP & PHATE
            sbs_c1, sbs_c2 = st.columns([2, 2])
            with sbs_c1:
                sbs_cscale = st.selectbox(
                    "Gradient Palette (for continuous metadata)",
                    options=CONTINUOUS_GRADIENTS,
                    index=CONTINUOUS_GRADIENTS.index(DEFAULT_CONTINUOUS_GRADIENT),
                    format_func=lambda g: f"{g} (Reverse RdGy)" if g == "RdGy_r" else g,
                    key="tab1_sbs_cscale",
                )
            with sbs_c2:
                sbs_pt_size = st.slider(
                    "Point Size",
                    min_value=1.0,
                    max_value=12.0,
                    value=5.0,
                    step=0.5,
                    key="tab1_sbs_pt_size",
                )

            c_u, c_p = st.columns(2)
            with c_u:
                st.markdown("#### UMAP")
                if "UMAP" not in avail_reds:
                    if st.button("⚡ Compute UMAP", key="sbs_compute_umap", type="primary"):
                        with st.spinner("Computing UMAP..."):
                            compute_umap(adata)
                        st.rerun()
                else:
                    fig_u = plot_embedding(
                        adata,
                        obsm_key=avail_reds["UMAP"],
                        color_by=color_meta if color_meta in adata.obs.columns else None,
                        color_scale=sbs_cscale,
                        point_size=sbs_pt_size,
                        title=f"UMAP — {color_meta}",
                    )
                    st.plotly_chart(fig_u, use_container_width=True)

            with c_p:
                st.markdown("#### PHATE")
                if "PHATE" not in avail_reds:
                    if st.button("⚡ Compute PHATE", key="sbs_compute_phate", type="primary"):
                        with st.spinner("Computing PHATE..."):
                            compute_phate(adata)
                        st.rerun()
                else:
                    fig_p = plot_embedding(
                        adata,
                        obsm_key=avail_reds["PHATE"],
                        color_by=color_meta if color_meta in adata.obs.columns else None,
                        color_scale=sbs_cscale,
                        point_size=sbs_pt_size,
                        title=f"PHATE — {color_meta}",
                    )
                    st.plotly_chart(fig_p, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Single-Gene Expression on Selected Dimensionality Reduction
    # -------------------------------------------------------------------------
    with tab_gene:
        st.subheader("Gene Expression on Dimensionality Reduction")
        st.caption("Select or enter a gene name to plot its expression across cells on UMAP, PHATE, or any reduction.")

        gene_list = list(adata.var_names)
        g_col1, g_col2 = st.columns([2, 2])
        with g_col1:
            selected_gene_dropdown = st.selectbox(
                "Search / Select Gene",
                options=gene_list,
                index=0,
                key="tab2_gene_select",
            )
        with g_col2:
            custom_gene_input = st.text_input(
                "Or Type Gene Symbol Directly (case-insensitive)",
                value="",
                placeholder="e.g. MS4A1, CD3D, LYZ...",
                key="tab2_gene_text",
            )

        target_gene_query = custom_gene_input.strip() if custom_gene_input.strip() else selected_gene_dropdown
        resolved_gene = resolve_gene_name(adata, target_gene_query)

        obsm_key_g, pt_size_g, cscale_g = render_reduction_controls(adata, key_prefix="tab2")

        if resolved_gene is None:
            st.error(f"Gene `{target_gene_query}` was not found in the dataset.")
        elif obsm_key_g is not None:
            expr_series = get_gene_expression(adata, resolved_gene)
            e_col1, e_col2 = st.columns([3, 1])
            with e_col1:
                fig_gene = plot_embedding(
                    adata,
                    obsm_key=obsm_key_g,
                    color_by=expr_series,
                    title=f"{resolved_gene} Expression ({obsm_key_g[2:].upper() if obsm_key_g.lower().startswith('x_') else obsm_key_g.upper()})",
                    color_scale=cscale_g,
                    point_size=pt_size_g,
                )
                st.plotly_chart(fig_gene, use_container_width=True)
            with e_col2:
                st.markdown(f"#### `{resolved_gene}` Summary")
                pct_expr = float((expr_series > 0).mean() * 100.0)
                st.metric("Expressing Cells (> 0)", f"{pct_expr:.1f}%")
                st.metric("Mean Expression", f"{float(expr_series.mean()):.3f}")
                st.metric("Max Expression", f"{float(expr_series.max()):.3f}")

                ct_col = st.session_state.get("meta_cell_type")
                if ct_col and ct_col in adata.obs.columns:
                    summary_df = (
                        pd.DataFrame({"Group": adata.obs[ct_col].astype(str), "Expr": expr_series.values})
                        .groupby("Group", observed=True)["Expr"]
                        .agg(Mean="mean", Pct_Expressed=lambda x: (x > 0).mean() * 100.0)
                        .round(2)
                        .sort_values("Mean", ascending=False)
                    )
                    st.markdown(f"**By `{ct_col}`:**")
                    st.dataframe(summary_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Multi-Gene Module Score Calculation & Plotting
    # -------------------------------------------------------------------------
    with tab_module:
        st.subheader("Multi-Gene Module Score")
        st.caption(
            "Provide multiple genes to compute a composite module score (via `scanpy.tl.score_genes` / Seurat `AddModuleScore` method) "
            "and visualize it across your selected dimensionality reduction."
        )

        default_genes_preview = list(adata.var_names[: min(4, len(adata.var_names))])
        m_col1, m_col2 = st.columns([2, 2])
        with m_col1:
            multi_selected = st.multiselect(
                "Select Genes from List",
                options=list(adata.var_names),
                default=default_genes_preview,
                key="tab3_multiselect",
            )
        with m_col2:
            pasted_genes = st.text_area(
                "Or Paste Gene List (comma, space, or newline separated)",
                value="",
                placeholder="e.g. CD3D, CD3E, IL7R, CCR7",
                height=100,
                key="tab3_textarea",
            )

        s_col1, s_col2 = st.columns([2, 1])
        with s_col1:
            score_name = st.text_input(
                "Module Score Name",
                value="Custom_Module_Score",
                key="tab3_score_name",
            )
        with s_col2:
            st.write("")
            calc_btn = st.button("🧮 Calculate Module Score", type="primary", use_container_width=True)

        # Combine multiselect + pasted genes
        parsed_pasted = [g for g in re.split(r"[\s,;]+", pasted_genes.strip()) if g] if pasted_genes.strip() else []
        combined_genes = list(dict.fromkeys(list(multi_selected) + parsed_pasted))

        if calc_btn or (score_name not in adata.obs.columns and combined_genes):
            if not combined_genes:
                st.warning("Please select or enter at least one gene.")
            else:
                try:
                    score_series, matched, missing = calculate_module_score(
                        adata,
                        genes=combined_genes,
                        score_name=score_name,
                    )
                    st.session_state["module_scores"][score_name] = {
                        "matched": matched,
                        "missing": missing,
                    }
                except ValueError as exc:
                    st.error(str(exc))

        if score_name in adata.obs.columns:
            info = st.session_state["module_scores"].get(score_name, {})
            matched_g = info.get("matched", combined_genes)
            missing_g = info.get("missing", [])
            st.success(
                f"**{score_name}** computed using **{len(matched_g)}** matched genes: `{', '.join(matched_g)}`"
                + (f"  \n⚠️ Not found ({len(missing_g)}): `{', '.join(missing_g)}`" if missing_g else "")
            )

            obsm_key_m, pt_size_m, cscale_m = render_reduction_controls(adata, key_prefix="tab3")
            if obsm_key_m is not None:
                fig_mod = plot_embedding(
                    adata,
                    obsm_key=obsm_key_m,
                    color_by=adata.obs[score_name],
                    title=f"{score_name} ({obsm_key_m[2:].upper() if obsm_key_m.lower().startswith('x_') else obsm_key_m.upper()})",
                    color_scale=cscale_m,
                    point_size=pt_size_m,
                )
                st.plotly_chart(fig_mod, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Violin Plots (by Sample, Condition, and Annotated Cell Type)
    # -------------------------------------------------------------------------
    with tab_violin:
        st.subheader("Single-Gene Violin Plots")
        st.caption(
            "Plot gene expression distributions across samples, conditions, and annotated cell types, "
            "with flexible metadata coloring options."
        )

        v_col1, v_col2, v_col3, v_col4 = st.columns([2, 2, 1.5, 1.5])
        with v_col1:
            v_gene_select = st.selectbox(
                "Select Gene",
                options=list(adata.var_names),
                index=0,
                key="tab4_gene_select",
            )
        with v_col2:
            v_gene_text = st.text_input(
                "Or Type Gene Name",
                value="",
                placeholder="e.g. ISG15, MS4A1, CD3D...",
                key="tab4_gene_text",
            )
        with v_col3:
            show_box = st.checkbox("Show Internal Box Plot", value=True, key="tab4_box")
        with v_col4:
            points_mode = st.selectbox(
                "Show Points",
                options=["outliers", "all", False],
                format_func=lambda x: "Outliers" if x == "outliers" else ("All Cells" if x == "all" else "Hidden"),
                index=0,
                key="tab4_points",
            )

        v_query = v_gene_text.strip() if v_gene_text.strip() else v_gene_select
        v_resolved = resolve_gene_name(adata, v_query)

        if v_resolved is None:
            st.error(f"Gene `{v_query}` was not found in the dataset.")
        else:
            v_expr = get_gene_expression(adata, v_resolved)
            cat_cols = get_categorical_columns(adata)
            avail_meta = cat_cols if cat_cols else list(adata.obs.columns)

            if not avail_meta:
                st.warning("No metadata columns available for grouping violin plots.")
            else:
                sample_col = st.session_state.get("meta_sample", avail_meta[0])
                cond_col = st.session_state.get("meta_condition", avail_meta[0])
                ct_col = st.session_state.get("meta_cell_type", avail_meta[0])

                # 1. Violin plot by Sample with selectable metadata coloring (Requirement 7)
                st.markdown("### 1. Expression by Sample")
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    sample_x = st.selectbox(
                        "Sample Column (X-axis)",
                        options=avail_meta,
                        index=avail_meta.index(sample_col) if sample_col in avail_meta else 0,
                        key="v_sample_x",
                    )
                with s_c2:
                    sample_color = st.selectbox(
                        "Color Violins By Metadata Column",
                        options=avail_meta,
                        index=avail_meta.index(sample_col) if sample_col in avail_meta else 0,
                        key="v_sample_color",
                    )

                # Resolve discrete color map for sample_color (use Dark2 if coloring by sample_x)
                sample_color_map = resolve_discrete_color_map(adata, sample_color)
                if sample_color == sample_x and sample_color_map is None:
                    sample_color_map = build_category_color_map(
                        adata.obs[sample_x].astype(str).unique(), "Dark2"
                    )

                fig_v_sample = plot_gene_violin(
                    adata,
                    expr_series=v_expr,
                    x_col=sample_x,
                    color_col=sample_color,
                    color_discrete_map=sample_color_map,
                    box=show_box,
                    points=points_mode,
                    title=f"{v_resolved} Expression by Sample ({sample_x}) — Colored by {sample_color}",
                )
                st.plotly_chart(fig_v_sample, use_container_width=True)

                st.markdown("---")

                # 2. Violin plots by Condition and Annotated Cell Type (Requirement 8)
                st.markdown("### 2. Expression by Condition & Annotated Cell Type")
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    cond_x = st.selectbox(
                        "Condition Column",
                        options=avail_meta,
                        index=avail_meta.index(cond_col) if cond_col in avail_meta else 0,
                        key="v_cond_x",
                    )
                with cc2:
                    ct_x = st.selectbox(
                        "Annotated Cell Type Column",
                        options=avail_meta,
                        index=avail_meta.index(ct_col) if ct_col in avail_meta else 0,
                        key="v_ct_x",
                    )
                with cc3:
                    extra_color_options = ["(Same as X-axis)"] + avail_meta
                    extra_color = st.selectbox(
                        "Optional Sub-grouping / Color Column",
                        options=extra_color_options,
                        index=0,
                        key="v_extra_color",
                    )

                plot_c1, plot_c2 = st.columns(2)
                with plot_c1:
                    color_for_cond = cond_x if extra_color == "(Same as X-axis)" else extra_color
                    cond_cmap = resolve_discrete_color_map(adata, color_for_cond)
                    if color_for_cond == cond_x and cond_cmap is None:
                        cond_cmap = build_category_color_map(
                            adata.obs[cond_x].astype(str).unique(), "Accent"
                        )
                    fig_v_cond = plot_gene_violin(
                        adata,
                        expr_series=v_expr,
                        x_col=cond_x,
                        color_col=color_for_cond,
                        color_discrete_map=cond_cmap,
                        box=show_box,
                        points=points_mode,
                        title=f"{v_resolved} by Condition ({cond_x})",
                    )
                    st.plotly_chart(fig_v_cond, use_container_width=True)

                with plot_c2:
                    color_for_ct = ct_x if extra_color == "(Same as X-axis)" else extra_color
                    ct_cmap = resolve_discrete_color_map(adata, color_for_ct)
                    if color_for_ct == ct_x and ct_cmap is None:
                        ct_cmap = build_category_color_map(
                            adata.obs[ct_x].astype(str).unique(), "Paired"
                        )
                    fig_v_ct = plot_gene_violin(
                        adata,
                        expr_series=v_expr,
                        x_col=ct_x,
                        color_col=color_for_ct,
                        color_discrete_map=ct_cmap,
                        box=show_box,
                        points=points_mode,
                        title=f"{v_resolved} by Annotated Cell Type ({ct_x})",
                    )
                    st.plotly_chart(fig_v_ct, use_container_width=True)

                # Combined Condition x Cell Type grouped violin plot
                with st.expander("📊 View Combined Cell Type × Condition Grouped Violin Plot", expanded=True):
                    combined_cond_cmap = resolve_discrete_color_map(adata, cond_x) or build_category_color_map(
                        adata.obs[cond_x].astype(str).unique(), "Accent"
                    )
                    fig_v_combined = plot_gene_violin(
                        adata,
                        expr_series=v_expr,
                        x_col=ct_x,
                        color_col=cond_x,
                        color_discrete_map=combined_cond_cmap,
                        box=show_box,
                        points=points_mode,
                        title=f"{v_resolved} by Annotated Cell Type ({ct_x}) Split by Condition ({cond_x})",
                    )
                    st.plotly_chart(fig_v_combined, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: Spatial Scatter Pie & SpatialCellChat Communication Field Explorer
    # -------------------------------------------------------------------------
    with tab_spatial:
        st.subheader("🧭 Spatial Scatter Pie & SpatialCellChat Communication Explorer")
        st.caption(
            "Visualize spatial cell-type composition with Scatter Pie plots (filtering by region and cell types), "
            "load precomputed SpatialCellChat analysis objects (`.rds` / `.h5ad`), or compute Communication Field "
            "(`netVisual_CommunField`) and Communication Field Grid (`netVisual_CommunFieldGrid`) plots."
        )

        # Panel to attach or load a SpatialCellChat analysis object from directory
        with st.expander("📂 Load / Attach External SpatialCellChat Analysis Object (`.rds` or `.h5ad`)", expanded=False):
            sc_files_in_dir = list_sc_files(st.session_state["current_dir"], recursive=False)
            ext_col1, ext_col2, ext_col3 = st.columns([2.5, 1.5, 1.5])
            with ext_col1:
                ext_path_str = st.text_input(
                    "Path to SpatialCellChat `.rds` or `.h5ad` file",
                    value="",
                    placeholder="/path/to/spatial_cellchat_object.rds",
                    key="sp_ext_path_input",
                )
            with ext_col2:
                dir_file_names = ["(Select from current folder)"] + [f.name for f in sc_files_in_dir]
                picked_dir_file = st.selectbox(
                    "Or pick file in current folder",
                    options=dir_file_names,
                    index=0,
                    key="sp_ext_dir_select",
                )
            with ext_col3:
                st.write("")
                if st.button("📥 Load Analysis Object", use_container_width=True, key="sp_load_scc_btn"):
                    target_p = (
                        Path(ext_path_str).expanduser()
                        if ext_path_str.strip()
                        else (
                            Path(st.session_state["current_dir"]) / picked_dir_file
                            if picked_dir_file != "(Select from current folder)"
                            else None
                        )
                    )
                    if target_p is None or not target_p.exists():
                        st.error("Please provide or select a valid SpatialCellChat `.rds` or `.h5ad` file.")
                    else:
                        try:
                            # First try loading as a full SpatialCellChat dataset or attaching to current adata
                            raw_check = load_spatial_analysis_object(target_p, target_adata=adata)
                            # If cell count differs from current adata, load the RDS directly as primary dataset
                            if (
                                raw_check.get("coordinates") is not None
                                and len(raw_check["coordinates"]) != adata.n_obs
                            ):
                                _load_file_into_state(target_p)
                            st.success(f"Loaded SpatialCellChat analysis from `{target_p.name}`!")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not load SpatialCellChat object: {exc}")

        coords_df = get_spatial_coordinates(adata)
        if coords_df is None:
            st.warning(
                "This dataset does not currently have 2D spatial coordinates (`obsm['spatial']` or `x_cent`/`y_cent` columns). "
                "Load a spatial `.rds` / `.h5ad` dataset, attach a SpatialCellChat object above, or click **🧪 Generate Demo** in the sidebar."
            )
        else:
            cat_cols = get_categorical_columns(adata)
            avail_meta = cat_cols if cat_cols else list(adata.obs.columns)
            ct_default = st.session_state.get("meta_cell_type") or (avail_meta[0] if avail_meta else None)

            sp_subtab_pie, sp_subtab_comm = st.tabs(
                [
                    "🥧 Spatial Scatter Pie Plot",
                    "📡 SpatialCellChat Communication Field & Grid",
                ]
            )

            # -----------------------------------------------------------------
            # Subtab A: Spatial Scatter Pie Plot
            # -----------------------------------------------------------------
            with sp_subtab_pie:
                st.markdown("#### Spatial Scatter Pie Plot (`visualizePie`)")
                st.caption(
                    "Filter by tissue region / metadata group or spatial coordinate bounding box, "
                    "and select specific cell types to display in each spot's pie chart."
                )

                p_c1, p_c2, p_c3, p_c4 = st.columns([1.8, 1.8, 1.4, 1.4])
                with p_c1:
                    pie_ct_col = st.selectbox(
                        "Cell Type Column",
                        options=avail_meta,
                        index=avail_meta.index(ct_default) if ct_default in avail_meta else 0,
                        key="pie_ct_col",
                    )
                with p_c2:
                    region_candidates = ["(All Regions / No Metadata Filter)"] + avail_meta
                    default_reg_idx = (
                        region_candidates.index("region")
                        if "region" in region_candidates
                        else (
                            region_candidates.index(st.session_state["meta_sample"])
                            if st.session_state.get("meta_sample") in region_candidates
                            else 0
                        )
                    )
                    pie_region_col = st.selectbox(
                        "Region / Slice Metadata Column",
                        options=region_candidates,
                        index=default_reg_idx,
                        key="pie_region_col",
                    )
                with p_c3:
                    grid_bins_pie = st.slider(
                        "Spatial Grid Binning (0 = per spot)",
                        min_value=0,
                        max_value=30,
                        value=0,
                        step=1,
                        help="Set > 0 to aggregate single cells into a 2D spatial grid (like makeGridSpatialCellChat) before plotting pies.",
                        key="pie_grid_bins",
                    )
                with p_c4:
                    pie_radius_scale = st.slider(
                        "Pie Radius Scale",
                        min_value=0.2,
                        max_value=2.5,
                        value=0.85,
                        step=0.05,
                        key="pie_radius_scale",
                    )

                # Region values filter + coordinate bounding box filter
                r_c1, r_c2, r_c3 = st.columns([2, 1.5, 1.5])
                if pie_region_col != "(All Regions / No Metadata Filter)" and pie_region_col in adata.obs.columns:
                    all_reg_vals = sorted(adata.obs[pie_region_col].astype(str).dropna().unique().tolist())
                    with r_c1:
                        selected_regions = st.multiselect(
                            f"Select Specific Regions (`{pie_region_col}`)",
                            options=all_reg_vals,
                            default=all_reg_vals,
                            key="pie_selected_regions",
                        )
                    reg_mask = adata.obs[pie_region_col].astype(str).isin(selected_regions).to_numpy()
                else:
                    with r_c1:
                        st.info("Showing all metadata regions. Choose a Region Column above to filter by region labels.")
                    reg_mask = np.ones(adata.n_obs, dtype=bool)

                x_min, x_max = float(coords_df["x"].min()), float(coords_df["x"].max())
                y_min, y_max = float(coords_df["y"].min()), float(coords_df["y"].max())
                with r_c2:
                    x_range_sel = st.slider(
                        "Spatial X Range Filter",
                        min_value=x_min,
                        max_value=x_max,
                        value=(x_min, x_max),
                        key="pie_x_range",
                    )
                with r_c3:
                    y_range_sel = st.slider(
                        "Spatial Y Range Filter",
                        min_value=y_min,
                        max_value=y_max,
                        value=(y_min, y_max),
                        key="pie_y_range",
                    )

                bbox_mask = (
                    (coords_df["x"].to_numpy() >= x_range_sel[0])
                    & (coords_df["x"].to_numpy() <= x_range_sel[1])
                    & (coords_df["y"].to_numpy() >= y_range_sel[0])
                    & (coords_df["y"].to_numpy() <= y_range_sel[1])
                )
                combined_spot_mask = reg_mask & bbox_mask

                if not np.any(combined_spot_mask):
                    st.warning("No spatial spots match the selected region / coordinate filters.")
                else:
                    sub_adata = adata[combined_spot_mask]
                    props_df, pie_coords_df = get_cell_type_proportions(
                        sub_adata,
                        cell_type_col=pie_ct_col,
                        grid_bins=grid_bins_pie,
                    )

                    all_pie_types = list(props_df.columns)
                    ct_sel_c1, ct_sel_c2, ct_sel_c3 = st.columns([3, 1, 1])
                    with ct_sel_c1:
                        selected_pie_cts = st.multiselect(
                            "Select Specific Cell Types for Scatter Pie",
                            options=all_pie_types,
                            default=all_pie_types,
                            key="pie_selected_cts",
                        )
                    with ct_sel_c2:
                        renorm_pies = st.checkbox(
                            "Renormalize selected types to 100%",
                            value=True,
                            key="pie_renorm",
                        )
                    with ct_sel_c3:
                        invert_pie_y = st.checkbox(
                            "Invert Y-axis (Image coords)",
                            value=False,
                            key="pie_invert_y",
                        )

                    pie_cmap = resolve_discrete_color_map(adata, pie_ct_col) or build_category_color_map(
                        all_pie_types, "Paired"
                    )
                    if not selected_pie_cts:
                        st.warning("Please select at least one cell type.")
                    else:
                        fig_pie = plot_scatter_pie(
                            proportions_df=props_df,
                            coords_df=pie_coords_df,
                            selected_cell_types=selected_pie_cts,
                            color_map=pie_cmap,
                            radius_scale=pie_radius_scale,
                            renormalize_selected=renorm_pies,
                            invert_y=invert_pie_y,
                            title=(
                                f"Spatial Scatter Pie — {len(pie_coords_df)} {'Grid Tiles' if grid_bins_pie > 0 else 'Spots'} "
                                f"({len(selected_pie_cts)}/{len(all_pie_types)} Cell Types)"
                            ),
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

            # -----------------------------------------------------------------
            # Subtab B: SpatialCellChat Communication Field & Grid
            # -----------------------------------------------------------------
            with sp_subtab_comm:
                st.markdown(
                    "#### SpatialCellChat Communication Field (`netVisual_CommunField`) & Grid (`netVisual_CommunFieldGrid`)"
                )

                # Expander to compute a new communication field on-demand from Ligand/Receptor genes
                with st.expander("➕ Compute New Spatial Communication Field from Ligand–Receptor Genes", expanded=False):
                    lr_c1, lr_c2, lr_c3, lr_c4 = st.columns([2, 2, 1.5, 1.2])
                    with lr_c1:
                        lig_sel = st.multiselect(
                            "Ligand Gene(s)",
                            options=list(adata.var_names),
                            default=[g for g in ["CXCL10", "CCL5", "HLA-DRA"] if g in adata.var_names][:2],
                            key="sp_lig_genes",
                        )
                    with lr_c2:
                        rec_sel = st.multiselect(
                            "Receptor Gene(s)",
                            options=list(adata.var_names),
                            default=[g for g in ["CCR7", "CD4", "CD8A"] if g in adata.var_names][:2],
                            key="sp_rec_genes",
                        )
                    with lr_c3:
                        new_sig_name = st.text_input(
                            "Pathway / Pair Name",
                            value="Custom_Signaling",
                            key="sp_new_sig_name",
                        )
                    with lr_c4:
                        st.write("")
                        if st.button("⚡ Compute Field", type="primary", use_container_width=True, key="sp_calc_lr_btn"):
                            if not lig_sel or not rec_sel:
                                st.warning("Select at least one ligand gene and one receptor gene.")
                            else:
                                with st.spinner(f"Computing communication field `{new_sig_name}`..."):
                                    compute_communication_field_from_lr_genes(
                                        adata,
                                        ligand_genes=lig_sel,
                                        receptor_genes=rec_sel,
                                        signaling_name=new_sig_name.strip() or "Custom_Signaling",
                                    )
                                st.rerun()

                avail_sig_by_slot = get_available_communication_signaling(adata)
                if not avail_sig_by_slot:
                    st.info(
                        "No precomputed SpatialCellChat communication fields found in this dataset yet. "
                        "Load a SpatialCellChat `.rds` object above, or open **➕ Compute New Spatial Communication Field from Ligand–Receptor Genes** to compute one on-demand."
                    )
                else:
                    cf_c1, cf_c2, cf_c3, cf_c4 = st.columns([1.3, 1.8, 1.4, 2.0])
                    slot_options = list(avail_sig_by_slot.keys())
                    with cf_c1:
                        sel_slot = st.selectbox(
                            "Slot (`netP` / `net`)",
                            options=slot_options,
                            index=0,
                            key="cf_slot_select",
                        )
                    sig_options = avail_sig_by_slot[sel_slot]
                    with cf_c2:
                        sel_sig = st.selectbox(
                            "Signaling Pathway / LR Pair",
                            options=sig_options,
                            index=0,
                            key="cf_sig_select",
                        )
                    with cf_c3:
                        sel_pattern = st.selectbox(
                            "Direction Pattern",
                            options=["outgoing", "incoming"],
                            format_func=lambda s: f"{s.capitalize()} ({'Sources' if s == 'outgoing' else 'Targets'})",
                            index=0,
                            key="cf_pattern_select",
                        )
                    with cf_c4:
                        cf_display_mode = st.radio(
                            "View Mode",
                            options=[
                                "Side-by-Side (Field & Grid)",
                                "Communication Field (Cell-level)",
                                "Communication Field Grid (Flow)",
                            ],
                            horizontal=True,
                            key="cf_display_mode",
                        )

                    cf_p1, cf_p2, cf_p3, cf_p4, cf_p5 = st.columns(5)
                    with cf_p1:
                        cf_arrow_scale = st.slider(
                            "Arrow Length Scale",
                            min_value=0.2,
                            max_value=3.0,
                            value=1.0,
                            step=0.1,
                            key="cf_arrow_scale",
                        )
                    with cf_p2:
                        cf_grid_bins = st.slider(
                            "Grid Resolution (bins)",
                            min_value=5,
                            max_value=35,
                            value=16,
                            step=1,
                            key="cf_grid_bins",
                        )
                    with cf_p3:
                        cf_grid_agg = st.selectbox(
                            "Grid Aggregation (`grid.field.mean`)",
                            options=["median", "sum"],
                            index=0,
                            key="cf_grid_agg",
                        )
                    with cf_p4:
                        cf_min_pct = st.slider(
                            "Min Magnitude Filter (%)",
                            min_value=0,
                            max_value=80,
                            value=10,
                            step=5,
                            key="cf_min_pct",
                        )
                    with cf_p5:
                        cf_invert_y = st.checkbox(
                            "Invert Y-axis (SpatialCellChat default)",
                            value=True,
                            key="cf_invert_y",
                        )

                    cf_label_col = st.session_state.get("meta_cell_type") or (avail_meta[0] if avail_meta else None)
                    cf_cmap = resolve_discrete_color_map(adata, cf_label_col)

                    if cf_display_mode == "Communication Field (Cell-level)":
                        fig_cf = plot_communication_field(
                            adata,
                            signaling=sel_sig,
                            slot_name=sel_slot,
                            pattern=sel_pattern,
                            label_col=cf_label_col,
                            color_map=cf_cmap,
                            arrow_scale=cf_arrow_scale,
                            min_mag_percentile=float(cf_min_pct),
                            invert_y=cf_invert_y,
                        )
                        st.plotly_chart(fig_cf, use_container_width=True)
                    elif cf_display_mode == "Communication Field Grid (Flow)":
                        fig_cfg = plot_communication_field_grid(
                            adata,
                            signaling=sel_sig,
                            slot_name=sel_slot,
                            pattern=sel_pattern,
                            grid_bins=cf_grid_bins,
                            grid_agg=cf_grid_agg,
                            label_col=cf_label_col,
                            color_map=cf_cmap,
                            arrow_scale=cf_arrow_scale,
                            min_mag_percentile=float(cf_min_pct),
                            invert_y=cf_invert_y,
                        )
                        st.plotly_chart(fig_cfg, use_container_width=True)
                    else:
                        col_f, col_g = st.columns(2)
                        with col_f:
                            fig_cf = plot_communication_field(
                                adata,
                                signaling=sel_sig,
                                slot_name=sel_slot,
                                pattern=sel_pattern,
                                label_col=cf_label_col,
                                color_map=cf_cmap,
                                arrow_scale=cf_arrow_scale,
                                min_mag_percentile=float(cf_min_pct),
                                invert_y=cf_invert_y,
                            )
                            st.plotly_chart(fig_cf, use_container_width=True)
                        with col_g:
                            fig_cfg = plot_communication_field_grid(
                                adata,
                                signaling=sel_sig,
                                slot_name=sel_slot,
                                pattern=sel_pattern,
                                grid_bins=cf_grid_bins,
                                grid_agg=cf_grid_agg,
                                label_col=cf_label_col,
                                color_map=cf_cmap,
                                arrow_scale=cf_arrow_scale,
                                min_mag_percentile=float(cf_min_pct),
                                invert_y=cf_invert_y,
                            )
                            st.plotly_chart(fig_cfg, use_container_width=True)


if __name__ == "__main__":
    main()
