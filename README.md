# Web-GIS-AI — Master Plan Copilot

AI-assisted spatial planning for town-development master planning. You write a
planning brief in plain English; the system produces a first-pass master plan —
suitability analysis, land-use allocation, road hierarchy, parcel layout,
facility siting, infrastructure and development-control checks — and exports it
as GIS, CAD, BIM and a printable plan set.

The guiding principle: **the language model never draws geometry.** It reads the
brief, selects tools, sets parameters and explains results. Every polygon, plot
boundary and road centreline is produced by deterministic, unit-tested code — so
the output is something a professional planner can review, adjust and sign.

## Architecture

- **`engine/` — `planeng`, the planning engine (headless, GIS-neutral).**
  A Python package where the spatial work happens: terrain and constraint
  analysis, multi-criteria suitability, land allocation, road and parcel
  generation, facility siting, infrastructure networks, development control,
  and exporters. This is the product; everything else is a front end over it.
- **`streamlit_app.py` — hosted demo front end.**
  A Streamlit app over the same engine, so a reviewer can be sent a link
  instead of installing anything. It calls the exact same `generate_master_plan`,
  `compare` and `export_all` entry points as the web app, so the two can never
  disagree about what a plan is.
- **Agentic AI layer.** Language-model agents read the brief and orchestrate the
  engine. Providers are pluggable: a hosted model via Groq, a local model via
  Ollama, or a deterministic rule-based reader as a fallback when no key is set.

## Project layout

```
Web-GIS-AI/
├── engine/                 # planeng — the planning engine
│   ├── src/planeng/
│   │   ├── analysis/       # terrain, constraints, suitability
│   │   ├── allocate/       # land-use allocation
│   │   ├── cadastre/       # block and parcel subdivision
│   │   ├── facilities/     # social-facility siting
│   │   ├── infrastructure/ # water, sewer, stormwater, power, lighting
│   │   ├── control/        # development-control rules
│   │   ├── agents/         # brief reader + LLM providers (groq, ollama)
│   │   ├── export/         # geojson, cad, bim, plans, schedules
│   │   └── api.py          # generate_master_plan / compare / export_all
│   └── pyproject.toml
├── streamlit_app.py        # hosted demo front end
├── requirements.txt        # deps for the hosted demo
├── runtime.txt             # Python version for the host
├── .streamlit/             # Streamlit config + secrets template
└── DEPLOY.md               # how to put the demo online
```

## Quick start

### Run the engine

```bash
cd engine
pip install -e .
planeng --help
```

### Run the hosted demo locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`).

## Configuration

The brief reader uses an LLM provider, configured through environment variables
(or Streamlit secrets). Copy the example and fill in your own key — the real
files are gitignored and must never be committed.

```bash
cp .env.example .env
# or, for the Streamlit demo:
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | API key for the hosted Groq model |
| `PLANENG_LLM_PROVIDER` | `groq` (hosted) or `ollama` (local) |

If no key is configured, the engine falls back to its rule-based brief reader and
the app says so rather than failing.

## Deployment

See [DEPLOY.md](DEPLOY.md) for putting the Streamlit demo online via Streamlit
Community Cloud.

## License

Released under the MIT License. See the [LICENSE](LICENSE) file for details.
