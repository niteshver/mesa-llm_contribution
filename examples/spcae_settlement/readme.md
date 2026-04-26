# Mars Settlement Model

## Summary

This example turns the Mars colonization ODD you shared into a Mesa + Mesa-LLM simulation with an optional Mesa-Geo layer. Martian colonists are modeled as `LLMAgent` instances that decide whether to repair the habitat, produce life-support resources, mine minerals, or support nearby settlers. The settlement tracks shared food, water, air, waste, minerals, shipments from Earth, habitat accidents, and the psychological load created by stressors.

The environment uses the Mars point shapefile in `data/` to bias mineral richness and patch productivity across the grid. When the optional `mesa-geo` stack is available, the same point geometry is also exposed through a `GeoSpace` overlay for map-style inspection.

## What Is Included

- `model.py`
  Core Mars settlement logic, resource accounting, shipments, accidents, and data collection.
- `agent.py`
  `MartianAgent` for LLM-driven settlers and `StressorAgent` for non-LLM disruption events.
- `tools.py`
  Mars-specific tools for local surveying, life-support production, mineral mining, habitat repair, and social support.
- `app.py`
  Solara view with a grid settlement view, charts, and a Mesa-Geo overlay when available.

## Custom Tools

- `survey_local_sector`
  Summarizes the current sector, nearby colonists, settlement reserves, and active stressors.
- `produce_resource`
  Produces `food`, `water`, `air`, or processes `waste` using the current patch capacity and the best nearby partner.
- `mine_minerals`
  Extracts minerals from the current sector and converts part of the output into technology gains.
- `repair_habitat`
  Attempts to remove the strongest habitat accident stressor and recover lost resources.
- `support_neighbor`
  Improves coping capacity and health for a nearby colonist.

## LLM Setup

The default model is:

```python
llm_model="ollama_chat/llama3.2:latest"
```

To prepare Ollama locally:

```bash
ollama pull llama3.2:latest
ollama serve
```

If you use another provider, change `llm_model` in `app.py` and provide the matching API key in `.env`.

## Mesa-Geo Note

The repo currently only includes `MARS_nomenclature_center_pts.shp`. Many GIS tools expect the matching `.shx`, `.dbf`, and often `.prj` sidecar files too. To keep the example usable right now, `model.py` reads the point geometry directly from the raw `.shp` file and projects it onto the Mesa grid.

If you later add the missing sidecar files and install the geo stack, the app can also render the Mars reference points in a Mesa-Geo view.

## How To Run

From the project root:

```bash
pip install -e .
pip install -U --pre mesa-geo
cd examples/spcae_settlement
solara run app.py
```

## Suggested Next Steps

- Add the missing shapefile sidecars so you can recover crater or landmark attributes, not just geometry.
- Promote the Mars point density into named regions such as mining zones, safe habitat zones, or high-risk accident sectors.
- Add a second agent class for robotic support units so human colonists can choose between direct labor and supervising autonomous systems.
