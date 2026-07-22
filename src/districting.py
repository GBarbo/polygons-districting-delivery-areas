# %% [markdown]
# # Last-mile Districting for Osasco
#
# First version of the districting model described in the article.
# Cells are marked with `# %%` so the script can be executed in parts
# in a Jupyter-like interactive window (VS Code, Spyder, PyCharm).
#
# Pipeline: setor polygons + anonymised parcel volumes -> adjacency
# graph with edge weight = shared boundary length -> 2-layer GCN
# encoder + softmax head -> soft assignment S in [0,1]^{n x k} ->
# unsupervised loss (balance + barrier-aware cut + compactness) ->
# argmax + exclave repair -> K delivery districts.
#
# All figures used in the paper are saved to ../article/images/.

# %%
# Imports and configuration
from __future__ import annotations

import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd() / "src"
GEOJSON_PATH = HERE / "sp" / "sp.geojson"
VOLUME_PATH = HERE / "volume_anon.csv"       # pre-anonymised, see article/anonymize.py

IMAGES_DIR = HERE.parent / "article" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SETORES_PATH   = IMAGES_DIR / "osasco_setores.png"
ADJACENCY_PATH = IMAGES_DIR / "osasco_adjacency.png"
LOSS_PATH      = IMAGES_DIR / "osasco_loss_curves.png"
DISTRICTS_PATH = IMAGES_DIR / "osasco_districts.png"

CITY = "Osasco"
K = 15                        # number of delivery areas
PROJECTED_CRS = "EPSG:31983"  # SIRGAS 2000 / UTM 23S, distances in metres

# Loss weights (see article, "Loss Function"). Each term is normalised
# to be roughly O(1), so these weights directly control the trade-off:
#   W_BALANCE  -> how strictly volumes must be equal across districts
#   W_CUT      -> how strongly we avoid cutting long open borders
#                 (short bridge-like borders are cheap to cut, so this
#                 term is what makes the model barrier-aware)
#   W_COMPACT  -> how round / disk-like each district must be. Raising
#                 this pulls districts toward their centre and
#                 discourages elongated shapes, isthmuses and exclaves.
W_BALANCE = 1.0
W_CUT     = 0.3
W_COMPACT = 1.0

# Training
EPOCHS = 1000
LR = 5e-3
HIDDEN = 32
SEED = 0


# %%
# Function definitions
def load_city_polygons(city: str) -> gpd.GeoDataFrame:
    """Load setores censitarios for a municipality, projected in metres."""
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf = gdf[gdf["NM_MUN"] == city].copy()
    gdf = gdf.to_crs(PROJECTED_CRS)
    gdf["CD_SETOR"] = gdf["CD_SETOR"].astype(str)
    return gdf.reset_index(drop=True)


def load_volumes() -> pd.DataFrame:
    """Load already-anonymised parcel volumes per setor censitario."""
    return pd.read_csv(VOLUME_PATH, dtype={"CD_SETOR": str})


def merge_volumes(gdf: gpd.GeoDataFrame, vols: pd.DataFrame) -> gpd.GeoDataFrame:
    merged = gdf.merge(vols, on="CD_SETOR", how="left")
    merged["vol"] = merged["vol"].fillna(0.0)
    return merged


def build_adjacency(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Adjacency edges with weight = shared boundary length in metres."""
    n = len(gdf)
    sindex = gdf.sindex
    src: list[int] = []
    dst: list[int] = []
    weight: list[float] = []
    for i in tqdm(range(n), desc="adjacency"):
        geom_i = gdf.geometry.iloc[i]
        for j in sindex.query(geom_i, predicate="touches"):
            j = int(j)
            if j <= i:
                continue
            shared = geom_i.boundary.intersection(gdf.geometry.iloc[j].boundary)
            length = float(shared.length) if not shared.is_empty else 0.0
            if length <= 0.0:
                continue
            src.append(i)
            dst.append(j)
            weight.append(length)
    return np.array([src, dst], dtype=np.int64), np.array(weight, dtype=np.float32)


def node_features(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Standardised centroid coords + log-volume + log-area."""
    centroids = gdf.geometry.centroid
    xy = np.stack([centroids.x.to_numpy(), centroids.y.to_numpy()], axis=1)
    xy = (xy - xy.mean(axis=0)) / (xy.std(axis=0) + 1e-6)
    log_vol = np.log1p(gdf["vol"].to_numpy())
    log_vol = (log_vol - log_vol.mean()) / (log_vol.std() + 1e-6)
    log_area = np.log1p(gdf.geometry.area.to_numpy())
    log_area = (log_area - log_area.mean()) / (log_area.std() + 1e-6)
    return np.stack([xy[:, 0], xy[:, 1], log_vol, log_area], axis=1).astype(np.float32)


def normalized_adjacency(edge_index: np.ndarray, n: int) -> torch.Tensor:
    """Symmetric normalised adjacency D^{-1/2} (A + I) D^{-1/2}."""
    A = np.zeros((n, n), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    A[edge_index[1], edge_index[0]] = 1.0
    A += np.eye(n, dtype=np.float32)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg + 1e-6)
    return torch.from_numpy(A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :])


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return adj_norm @ self.lin(x)


class DistrictingGNN(nn.Module):
    """2-layer GCN encoder + softmax partitioning head (GAP-style)."""

    def __init__(self, in_dim: int, hidden: int, k: int):
        super().__init__()
        self.g1 = GCNLayer(in_dim, hidden)
        self.g2 = GCNLayer(hidden, hidden)
        self.head = nn.Linear(hidden, k)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.g1(x, adj_norm))
        h = F.relu(self.g2(h, adj_norm))
        return F.softmax(self.head(h), dim=-1)


def districting_loss(
    S: torch.Tensor,
    volumes: torch.Tensor,
    centroids_std: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the three loss terms (balance, cut, compact) individually.

    The three terms mirror one-to-one the article's ``Loss Function''
    subsection and are combined by the caller with user-chosen weights.
    """

    # ------------------------------------------------------------
    # (a) L_balance : volume equality across districts
    # ------------------------------------------------------------
    v_per_d = volumes @ S                              # (k,)
    v_target = volumes.sum() / S.shape[1]
    L_balance = ((v_per_d - v_target) ** 2).mean() / (v_target ** 2 + 1e-6)

    # ------------------------------------------------------------
    # (b) L_cut : barrier-aware boundary cost
    # ------------------------------------------------------------
    src, dst = edge_index[0], edge_index[1]
    disagree = 1.0 - (S[src] * S[dst]).sum(dim=-1)     # (E,)
    L_cut = (edge_weight * disagree).sum() / (edge_weight.sum() + 1e-6)

    # ------------------------------------------------------------
    # (c) L_compact : moment-of-inertia compactness
    # ------------------------------------------------------------
    mu = (S.T @ centroids_std) / (S.sum(dim=0, keepdim=True).T + 1e-6)  # (k, 2)
    diff = centroids_std.unsqueeze(1) - mu.unsqueeze(0)                 # (n, k, 2)
    L_compact = (S * (diff ** 2).sum(dim=-1)).sum() / S.shape[0]

    return L_balance, L_cut, L_compact


def decode_and_repair(S: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Argmax decoding followed by exclave-repair."""
    labels = S.argmax(axis=1).astype(np.int64)
    n = len(labels)
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(edge_index[0], edge_index[1]):
        adj[a].append(int(b))
        adj[b].append(int(a))

    changed = True
    while changed:
        changed = False
        for d in range(int(labels.max()) + 1):
            nodes_d = np.where(labels == d)[0]
            if len(nodes_d) == 0:
                continue
            visited: set[int] = set()
            components: list[list[int]] = []
            for u in nodes_d:
                if u in visited:
                    continue
                stack, comp = [int(u)], []
                while stack:
                    v = stack.pop()
                    if v in visited:
                        continue
                    visited.add(v)
                    comp.append(v)
                    for w in adj[v]:
                        if labels[w] == d and w not in visited:
                            stack.append(w)
                components.append(comp)
            if len(components) <= 1:
                continue
            # keep the largest component; reassign the rest to the
            # most common foreign neighbour label
            components.sort(key=len, reverse=True)
            for small in components[1:]:
                foreign = [labels[w] for v in small for w in adj[v] if labels[w] != d]
                if foreign:
                    labels[small] = max(set(foreign), key=foreign.count)
                    changed = True
    return labels


# %%
# 1. Load city polygons and merge volumes
gdf = load_city_polygons(CITY)
vols = load_volumes()
gdf = merge_volumes(gdf, vols)
print(f"{CITY}: {len(gdf)} setores, total volume = {gdf['vol'].sum():.0f}")


# %%
# 2. Overview of the setores censitarios of the target city
fig, ax = plt.subplots(figsize=(9, 9))
gdf.plot(
    ax=ax,
    facecolor="#e8f0ff",
    edgecolor="#3a5f9e",
    linewidth=0.4,
)
ax.set_title(f"{CITY}: IBGE setores censitarios")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(SETORES_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"saved: {SETORES_PATH}")


# %%
# 3. Build the adjacency graph
edge_index, edge_weight = build_adjacency(gdf)
print(f"adjacency: {edge_index.shape[1]} edges, "
      f"weight in [{edge_weight.min():.1f}, {edge_weight.max():.1f}] m")


# %%
# 4. Adjacency-graph visualisation (line width proportional to shared
#    boundary length, so barrier-like short edges appear thin)
centroids = gdf.geometry.centroid
cx = centroids.x.to_numpy()
cy = centroids.y.to_numpy()

fig, ax = plt.subplots(figsize=(9, 9))
gdf.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.2)
w_scaled = 0.15 + 1.6 * (edge_weight / edge_weight.max())  # visible line widths
for (i, j), w in zip(edge_index.T, w_scaled):
    ax.plot([cx[i], cx[j]], [cy[i], cy[j]],
            color="black", linewidth=float(w), alpha=0.6, solid_capstyle="round")
ax.scatter(cx, cy, s=4, color="crimson", zorder=3)
ax.set_title(f"{CITY}: setor adjacency graph (edge width $\\propto$ shared border length)")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(ADJACENCY_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"saved: {ADJACENCY_PATH}")


# %%
# 5. Node features and normalised adjacency
X = node_features(gdf)
adj_norm = normalized_adjacency(edge_index, n=X.shape[0])


# %%
# 6. Train the GNN
#
# The total loss is an explicit weighted sum of three terms:
#
#     L_total = W_BALANCE * L_balance     # volume equality
#             + W_CUT     * L_cut         # barrier-aware cut cost
#             + W_COMPACT * L_compact     # moment-of-inertia compactness
#
# Each term is computed by ``districting_loss`` and tracked separately
# in ``history`` so we can inspect the individual contributions in the
# next cell.
torch.manual_seed(SEED)
model = DistrictingGNN(in_dim=X.shape[1], hidden=HIDDEN, k=K)
opt = torch.optim.Adam(model.parameters(), lr=LR)

x_t = torch.from_numpy(X)
volumes_t = torch.tensor(gdf["vol"].to_numpy(), dtype=torch.float32)
centroids_std = x_t[:, :2]
ei_t = torch.from_numpy(edge_index).long()
ew_t = torch.from_numpy(edge_weight)

history: dict[str, list[float]] = {
    "total":   [],
    "balance": [],   # L_balance,  weighted by W_BALANCE
    "cut":     [],   # L_cut,      weighted by W_CUT
    "compact": [],   # L_compact,  weighted by W_COMPACT
}

train_start = time.perf_counter()
pbar = tqdm(range(EPOCHS), desc="training")
for epoch in pbar:
    S = model(x_t, adj_norm)

    # --- the three loss terms, computed individually -------------
    L_balance, L_cut, L_compact = districting_loss(
        S, volumes_t, centroids_std, ei_t, ew_t
    )

    # --- explicit weighted combination ---------------------------
    L_total = (
        W_BALANCE * L_balance
        + W_CUT     * L_cut
        + W_COMPACT * L_compact
    )

    opt.zero_grad()
    L_total.backward()
    opt.step()

    history["total"].append(L_total.item())
    history["balance"].append(L_balance.item())
    history["cut"].append(L_cut.item())
    history["compact"].append(L_compact.item())

    pbar.set_postfix(
        total=f"{L_total.item():.4f}",
        balance=f"{L_balance.item():.4f}",
        cut=f"{L_cut.item():.4f}",
        compact=f"{L_compact.item():.4f}",
    )
train_elapsed = time.perf_counter() - train_start
print(f"training took {train_elapsed:.1f}s ({EPOCHS} epochs)")

with torch.no_grad():
    S_np = model(x_t, adj_norm).cpu().numpy()


# %%
# 7. Loss curves (per-term breakdown)
fig, ax = plt.subplots(figsize=(8, 4))
for name, series in history.items():
    ax.plot(series, label=name)
ax.set_xlabel("epoch")
ax.set_ylabel("loss value")
ax.set_title(f"Training losses ({EPOCHS} epochs, {train_elapsed:.1f}s)")
ax.legend()
plt.tight_layout()
plt.savefig(LOSS_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"saved: {LOSS_PATH}")


# %%
# 8. Decode + exclave repair, with per-district statistics
labels = decode_and_repair(S_np, edge_index)

# Assemble district-level geometry to measure shape compactness
gdf_with_labels = gdf.copy()
gdf_with_labels["district"] = labels
districts_gdf = gdf_with_labels.dissolve(by="district")
districts_gdf["area"] = districts_gdf.geometry.area
districts_gdf["perimeter"] = districts_gdf.geometry.length
# Polsby-Popper compactness: 1.0 = perfect circle, near 0 = elongated / spiky
districts_gdf["polsby_popper"] = (
    4.0 * np.pi * districts_gdf["area"] / (districts_gdf["perimeter"] ** 2)
)

v_target = gdf["vol"].sum() / int(labels.max() + 1)

print("\nfinal training losses (unweighted):")
print(f"  total   = {history['total'][-1]:.4f}")
print(f"  balance = {history['balance'][-1]:.4f}")
print(f"  cut     = {history['cut'][-1]:.4f}")
print(f"  compact = {history['compact'][-1]:.4f}")

print(f"\ntarget volume per district = {v_target:,.0f}\n")
print(f"{'d':>3} {'setores':>8} {'volume':>13} {'dev %':>7} {'PP':>6}")
abs_dev = []
for d in range(int(labels.max()) + 1):
    mask = labels == d
    vol_d = float(gdf["vol"][mask].sum())
    dev = 100.0 * (vol_d - v_target) / v_target
    abs_dev.append(abs(dev))
    pp = float(districts_gdf.loc[d, "polsby_popper"])
    print(f"{d:>3} {int(mask.sum()):>8} {vol_d:>13,.0f} {dev:>+7.1f} {pp:>6.3f}")

print("\naggregate summary")
print(f"  max |deviation|    : {max(abs_dev):.1f} %")
print(f"  mean |deviation|   : {sum(abs_dev) / len(abs_dev):.1f} %")
print(f"  Polsby-Popper min  : {districts_gdf['polsby_popper'].min():.3f}")
print(f"  Polsby-Popper median: {districts_gdf['polsby_popper'].median():.3f}")
print(f"  Polsby-Popper max  : {districts_gdf['polsby_popper'].max():.3f}")


# %%
# 9. Final districts map
gdf_plot = gdf.copy()
gdf_plot["district"] = labels

fig, ax = plt.subplots(figsize=(10, 10))
gdf_plot.plot(
    column="district",
    cmap="tab20",
    ax=ax,
    edgecolor="white",
    linewidth=0.3,
    categorical=True,
    legend=True,
    legend_kwds={"title": "District", "loc": "center left",
                 "bbox_to_anchor": (1.02, 0.5), "fontsize": 8, "ncol": 1},
)
ax.set_title(f"{CITY}: {K} balanced delivery districts")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(DISTRICTS_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"saved: {DISTRICTS_PATH}")

# %%
