# Polygons Districting for Delivery Areas

Giovanni Barboza - July 2026

A first-version tool that automatically divides a Brazilian municipality
into `k` balanced last-mile delivery districts using an unsupervised graph
neural network. IBGE *setores censitários* are used as the atomic building
blocks.

## What it does

Given IBGE setor geometries and per-setor parcel volumes, the pipeline:

1. Builds an adjacency graph where nodes are setores and edges connect
   pairs that share a physical boundary. Each edge is weighted by the
   length of that shared boundary.
2. Encodes the graph with a 2-layer GCN, producing a soft assignment of
   every setor to one of the `k` districts.
3. Optimises three unsupervised losses jointly:
   - **Balance** — equalises parcel volume across districts.
   - **Barrier-aware cut** — cheap to sever short edges (bridges over
     rivers, narrow crossings) and expensive to sever long open borders.
   - **Compactness** — pulls each district around its own centre,
     discouraging elongated shapes, isthmuses and exclaves.
4. Decodes the soft assignment via argmax and repairs any exclaves by
     reassigning them to the majority label of their neighbours.

## Repository layout

```
src/
├── districting.py     # cell-based pipeline script
├── volume_anon.csv    # anonymised per-setor parcel volumes
├── main.py            # small exploration script
└── sp/                # IBGE setor censitário geometry for SP state
```

## Requirements

- Python 3.10+
- `geopandas`, `pandas`, `numpy`, `matplotlib`
- `torch`
- `tqdm`

Install with pip:

```bash
pip install geopandas pandas numpy matplotlib torch tqdm
```

## Running

`src/districting.py` is organised as a sequence of `# %%` cells, so it can
be run end-to-end from the command line or stepped through interactively
in VS Code / Jupyter:

```bash
python src/districting.py
```

It loads the setor geometry and anonymised volumes, trains the GNN,
decodes the districts and writes the resulting figures (setor overview,
adjacency graph, loss curves, final districts) to the output directory
configured at the top of the script.

## Configuration

The main knobs live at the top of `src/districting.py`:

| Parameter | Meaning |
|---|---|
| `CITY` | Municipality name (default `Osasco`). |
| `K` | Number of delivery districts. |
| `EPOCHS`, `LR`, `HIDDEN` | Training hyperparameters. |
| `W_BALANCE`, `W_CUT`, `W_COMPACT` | Loss weights controlling the trade-off between volume equality, barrier awareness and district roundness. |
| `IMAGES_DIR` | Where the output figures are saved. |

## License

Research prototype. Use at your own risk.
