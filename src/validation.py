"""Cross-city validation of the Osasco districting pipeline.

Runs the same GCN + unsupervised loss used in ``districting.py`` on
four extra municipalities of the S\u00e3o Paulo metropolitan region,
with the operational ``k`` provided for each. Saves one district
map per city and a combined loss-curves figure into
``article/images/``, and prints a summary table used by the article's
``Validation`` section.

All model components and helpers are imported directly from
``districting.py`` so there is a single source of truth for the
pipeline.
"""
from __future__ import annotations

import time
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from districting import (
    EPOCHS,
    HIDDEN,
    LR,
    PROJECTED_CRS,
    SEED,
    W_BALANCE,
    W_COMPACT,
    W_CUT,
    DistrictingGNN,
    build_adjacency,
    decode_and_repair,
    districting_loss,
    merge_volumes,
    node_features,
    normalized_adjacency,
)

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd() / "src"
# Trimmed geojson containing Osasco + the four validation cities;
# same file used by ``districting.py``.
GEOJSON_PATH = HERE / "sp" / "sp.geojson"
VOLUME_PATH = HERE / "volume_anon.csv"

IMAGES_DIR = HERE.parent / "article" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LOSS_PATH = IMAGES_DIR / "validation_loss_curves.png"

# (city name in NM_MUN, k) pairs to validate on
CITIES: list[tuple[str, int]] = [
    ("Guarulhos", 28),
    ("S\u00e3o Bernardo do Campo", 17),
    ("Diadema", 8),
    ("Embu das Artes", 5),
]


def slugify(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return n.lower().replace(" ", "_")


def load_city_polygons(gdf_all: gpd.GeoDataFrame, city: str) -> gpd.GeoDataFrame:
    """Filter a preloaded state geodataframe down to a single city."""
    gdf = gdf_all[gdf_all["NM_MUN"] == city].copy()
    gdf = gdf.to_crs(PROJECTED_CRS)
    gdf["CD_SETOR"] = gdf["CD_SETOR"].astype(str)
    return gdf.reset_index(drop=True)


def run_city(
    gdf_all: gpd.GeoDataFrame,
    vols: pd.DataFrame,
    city: str,
    k: int,
) -> dict:
    print(f"\n=== {city} (k={k}) ===")
    gdf = load_city_polygons(gdf_all, city)
    gdf = merge_volumes(gdf, vols)
    n = len(gdf)
    total_vol = float(gdf["vol"].sum())
    print(f"  {n} setores, total volume = {total_vol:,.0f}")

    edge_index, edge_weight = build_adjacency(gdf)
    print(f"  {edge_index.shape[1]} edges")

    X = node_features(gdf)
    adj_norm = normalized_adjacency(edge_index, n=n)

    torch.manual_seed(SEED)
    model = DistrictingGNN(in_dim=X.shape[1], hidden=HIDDEN, k=k)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    x_t = torch.from_numpy(X)
    volumes_t = torch.tensor(gdf["vol"].to_numpy(), dtype=torch.float32)
    centroids_std = x_t[:, :2]
    ei_t = torch.from_numpy(edge_index).long()
    ew_t = torch.from_numpy(edge_weight)

    history = {"total": [], "balance": [], "cut": [], "compact": []}

    t0 = time.perf_counter()
    for _ in tqdm(range(EPOCHS), desc=f"  train {city}", leave=False):
        S = model(x_t, adj_norm)
        L_bal, L_cut, L_comp = districting_loss(S, volumes_t, centroids_std, ei_t, ew_t)
        L_tot = W_BALANCE * L_bal + W_CUT * L_cut + W_COMPACT * L_comp
        opt.zero_grad()
        L_tot.backward()
        opt.step()
        history["total"].append(L_tot.item())
        history["balance"].append(L_bal.item())
        history["cut"].append(L_cut.item())
        history["compact"].append(L_comp.item())
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        S_np = model(x_t, adj_norm).cpu().numpy()
    labels = decode_and_repair(S_np, edge_index)

    gdf_lab = gdf.copy()
    gdf_lab["district"] = labels
    districts_gdf = gdf_lab.dissolve(by="district")
    districts_gdf["area"] = districts_gdf.geometry.area
    districts_gdf["perimeter"] = districts_gdf.geometry.length
    districts_gdf["pp"] = (
        4.0 * np.pi * districts_gdf["area"] / (districts_gdf["perimeter"] ** 2)
    )

    v_target = total_vol / k
    vols_per_d = np.array(
        [float(gdf["vol"][labels == d].sum()) for d in range(k)]
    )
    dev = (vols_per_d - v_target) / max(v_target, 1e-9)
    abs_dev_pct = 100.0 * np.abs(dev)

    slug = slugify(city)
    map_path = IMAGES_DIR / f"{slug}_districts.png"
    fig, ax = plt.subplots(figsize=(9, 9))
    gdf_lab.plot(
        column="district",
        cmap="tab20",
        ax=ax,
        edgecolor="white",
        linewidth=0.3,
        categorical=True,
        legend=True,
        legend_kwds={
            "title": "District",
            "loc": "center left",
            "bbox_to_anchor": (1.02, 0.5),
            "fontsize": 8,
            "ncol": 1,
        },
    )
    ax.set_title(f"{city}: {k} balanced delivery districts")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {map_path}")

    return {
        "city": city,
        "slug": slug,
        "k": k,
        "n_setores": n,
        "n_edges": int(edge_index.shape[1]),
        "total_vol": total_vol,
        "elapsed_s": elapsed,
        "L_bal": history["balance"][-1],
        "L_cut": history["cut"][-1],
        "L_comp": history["compact"][-1],
        "L_total": history["total"][-1],
        "mean_abs_dev_pct": float(abs_dev_pct.mean()),
        "max_abs_dev_pct": float(abs_dev_pct.max()),
        "pp_min": float(districts_gdf["pp"].min()),
        "pp_median": float(districts_gdf["pp"].median()),
        "pp_max": float(districts_gdf["pp"].max()),
        "history": history,
    }


def main() -> None:
    print(f"Loading geojson from {GEOJSON_PATH} ...")
    gdf_all = gpd.read_file(GEOJSON_PATH)
    vols = pd.read_csv(VOLUME_PATH, dtype={"CD_SETOR": str})

    results = [run_city(gdf_all, vols, city, k) for city, k in CITIES]

    # Combined 2x2 loss curves figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, r in zip(axes.flat, results):
        for name in ("total", "balance", "cut", "compact"):
            ax.plot(r["history"][name], label=name, linewidth=1.2)
        ax.set_title(f"{r['city']} (k={r['k']}, {r['elapsed_s']:.0f}s)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss value")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(LOSS_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {LOSS_PATH}")

    # Summary table
    summary_cols = [
        "city", "k", "n_setores", "n_edges", "elapsed_s",
        "L_bal", "L_cut", "L_comp", "L_total",
        "mean_abs_dev_pct", "max_abs_dev_pct",
        "pp_min", "pp_median", "pp_max",
    ]
    df = pd.DataFrame([{c: r[c] for c in summary_cols} for r in results])
    csv_path = HERE / "validation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}")

    print("\n=== Summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
