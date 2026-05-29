# climate-conservation

**Climate network conservation analysis — temperature correlation networks, station graphs, and conservation of elevation under warming.**

Climate stations form a spatial network where temperature correlations define edges. Measures conservation of physical attributes (elevation) on this network. Tests the hypothesis that climate change decreases conservation of the altitude-temperature relationship.

## What This Gives You

- **Synthetic climate networks** — 50 stations with lat/lon/elevation, correlated temperature series
- **Warming simulation** — adjustable warming rate with Arctic amplification
- **Extreme events** — heat waves, El Niño Pacific patterns
- **Station loss** — simulate sensor failures and measure conservation impact
- **Conservation of elevation** — tracks how warming disrupts the altitude-temperature relationship
- **Multi-panel visualization** — station maps, temperature series, conservation over time

## Quick Start

```bash
pip install numpy scipy matplotlib networkx
python climate_conservation.py
```

Outputs go to `figures/` with PNG plots.

## How It Fits

Part of the SuperInstance ecosystem:

- **[lattice-climate](https://github.com/SuperInstance/lattice-climate)** — Statistical mechanics climate models
- **[regime-detection](https://github.com/SuperInstance/regime-detection)** — Climate regime change detection
- **climate-conservation** — Spectral climate analysis (this repo)

## License

MIT
