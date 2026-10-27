"""End-to-end tests for scExplore data loading, analysis, and plotting."""

from __future__ import annotations

from pathlib import Path
import tempfile

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
from scexplore.data_loader import list_sc_files, load_dataset
from scexplore.directory_picker import generate_demo_dataset
from scexplore.plotting import (
    DEFAULT_CONTINUOUS_GRADIENT,
    get_palette_hex_list,
    plot_embedding,
    plot_gene_violin,
    resolve_discrete_color_map,
)
from scexplore.spatial import (
    compute_communication_field_from_lr_genes,
    get_available_communication_signaling,
    get_cell_type_proportions,
    is_spatial_dataset,
    plot_communication_field,
    plot_communication_field_grid,
    plot_scatter_pie,
)


def test_end_to_end_workflow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        demo_file = generate_demo_dataset(tmp_path)
        assert demo_file.exists()

        files = list_sc_files(tmp_path)
        assert demo_file in files

        adata = load_dataset(demo_file)
        assert adata.n_obs == 600
        assert adata.n_vars == 120

        # Check metadata detection
        guessed = guess_metadata_columns(adata)
        assert guessed["sample"] == "sample"
        assert guessed["condition"] == "condition"
        assert guessed["cell_type"] == "cell_type"
        assert "sample" in get_categorical_columns(adata)

        # Check that color maps were created on first read with Paired, Dark2, and Accent
        assert "scexplore_color_maps" in adata.uns
        ct_cmap = resolve_discrete_color_map(adata, "cell_type")
        sample_cmap = resolve_discrete_color_map(adata, "sample")
        cond_cmap = resolve_discrete_color_map(adata, "condition")

        assert ct_cmap is not None
        assert list(ct_cmap.values()) == get_palette_hex_list("Paired", len(ct_cmap))
        assert sample_cmap is not None
        assert list(sample_cmap.values()) == get_palette_hex_list("Dark2", len(sample_cmap))
        assert cond_cmap is not None
        assert list(cond_cmap.values()) == get_palette_hex_list("Accent", len(cond_cmap))
        assert DEFAULT_CONTINUOUS_GRADIENT == "RdGy_r"

        # Check UMAP and PHATE
        umap_key = compute_umap(adata)
        phate_key = compute_phate(adata)
        avail = get_available_reductions(adata)
        assert "UMAP" in avail and avail["UMAP"] == umap_key
        assert "PHATE" in avail and avail["PHATE"] == phate_key

        # Check single gene expression
        gene = resolve_gene_name(adata, "ms4a1")
        assert gene == "MS4A1"
        expr = get_gene_expression(adata, gene)
        assert len(expr) == adata.n_obs

        # Check module score
        score, matched, missing = calculate_module_score(
            adata, ["CD3D", "CD3E", "IL7R", "NONEXISTENT_GENE"], score_name="T_Cell_Module"
        )
        assert len(score) == adata.n_obs
        assert matched == ["CD3D", "CD3E", "IL7R"]
        assert missing == ["NONEXISTENT_GENE"]

        # Check plotting functions
        fig_emb_meta = plot_embedding(adata, umap_key, color_by="cell_type")
        assert len(fig_emb_meta.data) > 0

        fig_emb_gene = plot_embedding(adata, phate_key, color_by=expr)
        assert len(fig_emb_gene.data) > 0

        fig_emb_mod = plot_embedding(adata, umap_key, color_by=score)
        assert len(fig_emb_mod.data) > 0

        fig_v_sample = plot_gene_violin(adata, expr, x_col="sample", color_col="condition")
        assert len(fig_v_sample.data) > 0

        fig_v_cond = plot_gene_violin(adata, expr, x_col="condition")
        assert len(fig_v_cond.data) > 0

        fig_v_ct = plot_gene_violin(adata, expr, x_col="cell_type", color_col="condition")
        assert len(fig_v_ct.data) > 0

        # Check spatial detection, Scatter Pie (with region and cell-type subsetting), and SpatialCellChat fields
        assert is_spatial_dataset(adata)
        reg_mask = adata.obs["region"].astype(str).isin(["T_Cell_Zone", "B_Follicle_Zone"]).to_numpy()
        sub_adata = adata[reg_mask]
        props_df, coords_df = get_cell_type_proportions(sub_adata, cell_type_col="cell_type", grid_bins=10)
        fig_pie = plot_scatter_pie(
            props_df,
            coords_df,
            selected_cell_types=["CD4 T", "CD8 T", "B Cells"],
            color_map=ct_cmap,
        )
        assert len(fig_pie.data) > 0

        avail_sig = get_available_communication_signaling(adata)
        assert "netP" in avail_sig and "CXCL_CCL" in avail_sig["netP"]

        fig_cf = plot_communication_field(adata, signaling="CXCL_CCL", slot_name="netP", pattern="outgoing")
        assert len(fig_cf.data) > 0

        fig_cfg = plot_communication_field_grid(
            adata, signaling="CXCL_CCL", slot_name="netP", pattern="incoming", grid_bins=12
        )
        assert len(fig_cfg.data) > 0

        # Test computing a new communication field on-demand
        out_v, in_v = compute_communication_field_from_lr_genes(
            adata, ["CD3D", "CD3E"], ["HLA-DRA"], signaling_name="Test_LR"
        )
        assert out_v.shape == (adata.n_obs, 2)
        assert in_v.shape == (adata.n_obs, 2)


if __name__ == "__main__":
    test_end_to_end_workflow()
    print("All end-to-end tests passed!")
