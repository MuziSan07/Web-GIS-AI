# AI-Powered Spatial Planning & Town Development Master Planning Platform
## Response to Client — Feasibility, Approach, Budget and Timeline

**Prepared:** 17 August 2026
**Status:** Indicative. Figures are pre-discovery estimates, not a fixed quotation.

---

## 0. Direct answer

**Yes — this is technically achievable, and yes, we can build it.**

Two things must be said plainly before anything else, because they change how we should structure the engagement:

**a) "Live in 4 weeks" is not achievable for the full platform. A working prototype on your actual 600 ha site is.**
A statutory-grade town planning platform is a 6–9 month build. What we *can* put in front of you in 4 weeks is a real, prompt-driven vertical slice running on your real site data: type a planning brief in plain English, and the system produces a suitability analysis, land-use allocation, road hierarchy, facility siting, parcel layout, a 2D web plan, a basic 3D model and exportable GIS layers. Not a mock-up — a working engine, with a deliberately narrow feature set.

You said you will invest once you see a viable prototype. We agree with that sequencing. We propose exactly that as the first paid phase.

**b) The AI does not produce an approvable plan on its own, and should not be sold as doing so.**
It produces a technically-competent first draft in hours instead of months, plus scenario comparison and automated compliance checking. A registered town planner still reviews, adjusts and signs. That is not a limitation of our build — it is a legal requirement, and it is also the strongest commercial argument for the product: it makes one planner as productive as a department.

---

## 1. Project approach — your question 8, answered first

You asked: are we building a new GIS from scratch, or using ArcGIS Pro / ArcGIS Urban APIs and building the automation and AI agents on top? You said you prefer the latter.

**We agree, and we recommend the latter — with one important refinement.**

We are not building a GIS. Nobody should. But we also should not build the planning intelligence *inside* ArcGIS, because that permanently welds your product to one vendor's licence.

### Recommended architecture — three layers

**Layer 1 — Planning Engine (headless, GIS-neutral). This is the product.**
A Python service stack where the actual spatial work happens:
- PostgreSQL + PostGIS — the single source of truth for all spatial data
- GDAL / GeoPandas / Shapely — geometry processing
- pgRouting / NetworkX — road hierarchy, accessibility, walking catchments
- OR-Tools / Pyomo — multi-objective land allocation and facility location optimisation
- Deterministic, unit-tested geoprocessing functions: parcel subdivision, block generation, road platform generation, setback enforcement, utility corridor routing

**Layer 2 — Agentic AI layer.**
Claude-based agents that read a natural-language brief and orchestrate Layer 1. Specialist agents broadly as you outlined: Brief, Remote Sensing, Suitability, Transportation, Residential Layout, Social Facilities, Infrastructure, Development Control, Optimisation, Reporting.

> **The single most important engineering decision in this project:** the LLM never draws geometry. It interprets intent, selects tools, sets parameters, sequences the workflow, explains results and flags conflicts. Every polygon, every plot boundary, every road centreline is produced by deterministic, testable code. This is the difference between a demo that impresses and a system a professional will put their stamp on.

**Layer 3 — Clients and publication.**
- Web planning portal (MapLibre / deck.gl for 2D, CesiumJS for 3D) — the planner's primary workspace
- **ArcGIS adapter** — publish plans, zoning and development-control data directly into ArcGIS Urban via the Urban API; feature services to ArcGIS Enterprise; file geodatabase export for ArcGIS Pro
- **QGIS adapter** — QGIS plugin exposing the same engine, for analysts without Esri seats
- CAD (DWG/DXF), IFC (BIM), CityGML/3D Tiles, PDF/A statutory plan sets

### Why this beats building directly inside ArcGIS

| | Engine-first (recommended) | Build inside ArcGIS |
|---|---|---|
| Vendor lock-in | None — Esri is an output target | Total |
| Licence cost per user | Only for those who need ArcGIS | Every user needs a seat |
| Automated testing of planning logic | Straightforward | Very difficult |
| Your ability to resell the platform | You own the IP outright | Constrained by Esri terms |
| Government departments on Esri | Fully served via adapter | Fully served |

You get everything you wanted from the ArcGIS route — including ArcGIS Urban as the stakeholder-facing plan review and development-control environment — without the engine being trapped there.

### Verified: ArcGIS Urban is a viable automation target

We checked this rather than assuming. **ArcGIS Urban API v1.0 is released and is GraphQL-based, and Esri explicitly documents automating the creation of plans and projects programmatically**, including zoning and overlay parameters. Your preferred approach is sound. We would use it for plan publication, zoning and development-control data, and stakeholder review workflows.

### Findings on your open-source research — three corrections that matter

Your research mostly checks out. Three things need correcting before they end up in a plan or a budget:

**PlanX is real, but it is Turkish planning law in software.**
The PlanX suite genuinely exists on the QGIS plugin repository, including the CAD toolset, settlement planning toolset and urban resilience modules. However, the PlanX-UIP master plan toolset automates *Uygulama Imar Plani* — Turkish 1/1000 implementation plans. Its eight stages produce Turkish regulatory artefacts: urban character tables, EK-2 tables, and DOP (land readjustment) ratios. The road platform generator, junction trim and facade segmentation tools are genuinely useful and transferable. The regulatory output is not. Also note: it is a single-author academic project (Dokuz Eylul University) with effectively no community adoption yet — 0 stars, 0 forks on GitHub.
**Treat PlanX as a source of proven algorithms and a reference implementation, not as a foundation to build on.**

**QCity is real but cannot be used for Botswana yet.**
QCity was released July 2026 by North Road — one of the firms on your contact list — built with the City of Canning in Perth. It does dwelling yield, commercial floorspace yield, parking yield and 2D/3D visualisation. But **it is explicitly Australia-only at present and has not reached stable release.** Broader regional support is planned but not delivered. Do not put it on the critical path.

**GPL-3.0 is a genuine commercial constraint.**
QGIS is GPL, and so is PlanX. If you build your platform *as a QGIS plugin*, you may be obliged to release your source code under GPL — which undermines selling the platform as licensed or SaaS software. Our engine-first architecture avoids this entirely: the engine is your proprietary code, and the QGIS plugin is a thin GPL client that talks to it over an API. **This alone justifies the architecture.**

VoxCity and CesiumJS remain good candidates for the 3D layer and we will evaluate both in Sprint 0.

---

## 2. Module register — completed, in development, remaining

Modules are grouped by where the work actually comes from. "Off-the-shelf" means mature open source we configure rather than write.

| # | Module | Source | Effort |
|---|---|---|---|
| 1 | Spatial data store (PostGIS schema, versioning, audit trail) | Off-the-shelf + schema design | Low |
| 2 | Data ingestion (DEM, imagery, cadastral, hydrology, roads) | Off-the-shelf (GDAL) + connectors | Low |
| 3 | Terrain & constraint analysis (slope, aspect, drainage, flood, wetland) | Off-the-shelf + config | Low |
| 4 | **Land suitability engine** (weighted multi-criteria per land use) | Build | Medium |
| 5 | **Botswana planning standards rules engine** | **Build — highest domain effort** | **High** |
| 6 | **Multi-objective land allocation optimiser** | Build (OR-Tools) | High |
| 7 | **Road hierarchy generator** (primary to local, junctions, reserves) | Build; PlanX algorithms as reference | High |
| 8 | **Block & parcel subdivision engine** (cadastral layout) | Build | High |
| 9 | Social facility siting (schools, clinics, police, fire — by population + catchment) | Build | Medium |
| 10 | Infrastructure networks (water, sewer, stormwater, electrical, telecoms, lighting) | Build | High |
| 11 | Waste, cemeteries, environmental reserves, utility corridors | Build | Medium |
| 12 | **Development control engine** (plot size, setbacks, FAR, coverage, density, compatibility) | Build | High |
| 13 | Scenario generation & scoring (compact / linear / clustered / low-cost) | Build | Medium |
| 14 | **AI agent orchestration layer** (all 9–10 specialist agents) | Build | High |
| 15 | Natural-language brief parser to structured planning brief | Build | Medium |
| 16 | Web planning portal — 2D | Build on MapLibre/deck.gl | Medium |
| 17 | 3D city model & visualisation | Cesium / VoxCity + build | Medium |
| 18 | ArcGIS Urban / Pro / Enterprise adapter | Build on Urban API v1.0 | Medium |
| 19 | QGIS plugin client | Build | Low |
| 20 | CAD (DWG/DXF) and BIM (IFC) export | Off-the-shelf + build | Medium |
| 21 | Automated report & statutory plan-set generation | Build | Medium |
| 22 | Cost estimation & phasing schedules | Build | Medium |
| 23 | Human-in-the-loop review, approval workflow, version control | Build | Medium |
| 24 | Auth, roles, multi-tenancy, audit | Off-the-shelf + config | Low |
| 25 | Multilingual voice & chat assistant | Build — separate workstream | Medium |

> **Internal note — resolve before sending.** The "completed / in development" split must reflect what is genuinely in our codebase today. No module is asserted as complete above, because this repository contains no prior work. Mark actual status against each row before this goes out — overstating readiness here is the fastest way to lose the account in month two.

---

## 3. Timeline

### Sprint 0 — Prototype (4 weeks) — meets your emergency deadline
| Week | Deliverable |
|---|---|
| 1 | Site data ingested; PostGIS live; terrain/constraint analysis on the real 600 ha; core Botswana standards encoded (plot sizes, road reserves, facility ratios, catchments) |
| 2 | Suitability engine + land allocation optimiser producing zoned land-use plan |
| 3 | Road hierarchy + block/parcel subdivision + facility siting |
| 4 | Prompt interface, web 2D plan, basic 3D, GIS/CAD export, **live demo account** |

**End state:** you type a brief, the system produces a first-pass master plan for your site, and you can inspect and export it. Demo account provided as requested.

**Deliberately excluded from Sprint 0:** full development control automation, all infrastructure networks, multi-scenario optimisation, statutory report generation, production hardening.

### Phase 1 — MVP (weeks 5–16)
All infrastructure networks, full development control engine, scenario generation and scoring, ArcGIS Urban integration, report generation, approval workflow.
**Exit criterion: one complete master plan for the 600 ha site, produced by the system and validated by a registered planner.**

### Phase 2 — Production platform (weeks 17–36)
Multi-project, multi-tenant, full 3D, BIM/CAD pipeline, cost and phasing modules, performance and security hardening, training, handover, on-premises deployment option.

### Voice assistant workstream (weeks 1–6, parallel)
Independent team; does not compete for GIS engineering capacity.

---

## 4. Budget summary (USD, indicative)

| Phase | Duration | Range |
|---|---|---|
| Sprint 0 — Prototype | 4 weeks | **55,000 – 85,000** |
| Phase 1 — MVP | 12 weeks | **180,000 – 300,000** |
| Phase 2 — Production platform | 20 weeks | **260,000 – 450,000** |
| **Cumulative to full production** | **~9 months** | **495,000 – 835,000** |
| Voice & multilingual assistant | 4–6 weeks | 40,000 – 70,000 |
| Annual run — cloud, LLM inference, support & SLA | per year | 75,000 – 140,000 |

**AI agent development specifically** (your question 5) — approximately **120,000 – 200,000** of the above. Worth being precise about what this money buys, because "training AI agents" is usually misunderstood:
- Agent design, tool definitions, orchestration and guardrails — the bulk of it
- Encoding Botswana planning standards into a machine-readable knowledge base — the highest-value, most labour-intensive part
- An evaluation harness that tests agent output against known-good plans, so quality is measured rather than assumed

Custom model fine-tuning is **probably unnecessary** — the work is tool design and domain knowledge encoding, not model training. If it proves necessary, budget a further 15,000 – 25,000.

**Excluded from our fees — client-borne:**
- Esri licensing (ArcGIS Urban, Pro, Enterprise) — indicative 30,000 – 90,000 per year depending on named users and deployment; must be quoted by Esri. This cost disappears entirely on the QGIS-only path.
- Satellite/aerial imagery acquisition, drone survey, geotechnical and hydrological studies
- Local professional fees for the registered planner and engineers (see section 6)

---

## 5. Preferred pricing model — hybrid

| Component | Model | Rationale |
|---|---|---|
| Sprint 0 and Phase 1 | **Fixed fee, milestone-based** | You get cost certainty; we carry delivery risk |
| Phase 2 | Fixed fee per module, or time-and-materials with a cap | Scope will evolve as planners use it |
| Ongoing platform | **Annual subscription** — per named planner seat plus per active project | Covers cloud, LLM inference, updates, support |
| Support and SLA | Tiered annual (Standard / Priority / Dedicated) | |
| Government / sovereign deployments | Perpetual licence plus annual maintenance (18–22%) | Public procurement usually needs capex, not opex |
| IP | You own the Botswana ruleset and your data outright. Core engine licensed to you perpetually, with a negotiable revenue share if you resell regionally | Aligns us with your growth |

**Recommendation:** fixed-fee delivery, then subscription. Avoid a pure once-off deal — this platform needs continuous rule updates, model upgrades and support, and a once-off structure leaves nobody funded to provide them.

---

## 6. Experts required

### Our side — software delivery (peak ~10 people)
Solution architect · 2x geospatial engineer (PostGIS, GDAL, spatial algorithms) · 2x AI/agent engineer · backend engineer · frontend and 3D engineer (Cesium/WebGL) · DevOps/cloud engineer · QA engineer with geospatial testing experience · project manager / business analyst

### Your side or locally contracted — domain authority
These are not optional, and several must be **registered to practise in Botswana** because they carry statutory sign-off:

| Discipline | Why we need them | Statutory |
|---|---|---|
| **Registered Town Planner (lead)** | Defines standards, validates every output, **signs the plan** | **Yes** |
| Civil / infrastructure engineer | Water, sewer, stormwater, road geometry and pavement | Yes |
| Electrical engineer | Reticulation, substations, street lighting | Yes |
| Transport planner | Road hierarchy, junctions, public transport, parking | |
| Environmental management specialist | EIA, wetlands, protected areas, sensitivity mapping | Yes |
| Land surveyor / cadastral engineer | Parcel geometry, beacons, deeds-registrable layouts | **Yes** |
| Hydrologist / geotechnical engineer | Flooding, drainage, ground conditions, developable land | |
| Urban economist / development finance analyst | Land budget, phasing, cost and revenue modelling | |
| Planning law advisor | Statutory approval route and compliance | |

**Critical:** the Registered Town Planner must be engaged from **week 1 of Sprint 0**, not brought in to review at the end. If the standards encoded in week 1 are wrong, everything built on them is wrong. This is the single highest-risk dependency in the project.

---

## 7. Deployment plan

**Recommended: cloud-first, with an on-premises path for sovereign data.**

**Cloud (primary)** — Azure South Africa North (Johannesburg) or AWS af-south-1 (Cape Town). These are the nearest regions to Gaborone; they keep data in-continent for residency and give acceptable latency.
- Containerised services on managed Kubernetes
- Managed PostgreSQL + PostGIS, with read replicas
- Object storage for rasters, imagery and 3D tiles; CDN for tile delivery
- Optional GPU nodes for 3D tiling and heavy raster processing
- LLM via API with regional routing; all prompts and outputs logged for audit
- Full infrastructure-as-code (Terraform), automated backup, DR and monitoring

**On-premises (option)** — for government data sovereignty requirements:
- Full containerised install on client hardware at the department
- Cadastral and survey data never leaves the premises
- **Honest caveat:** a fully air-gapped deployment requires a self-hosted open-weight model, and plan quality will be measurably lower than with a frontier model. We will quantify that gap during Phase 1 so the decision is made on evidence rather than assumption.

**Recommended middle path — hybrid:** engine and portal in cloud; master cadastral data replicated to an on-premises PostGIS instance at the department over VPN, so government retains a sovereign copy of record.

**Environments:** Development, Staging, Production, plus a permanent demo tenant for stakeholder presentations.

---

## 8. What we need from you — this is the critical path

Sprint 0 stalls in week 1 without these. Please treat this as the most urgent item in this document:

**Site data**
1. Site boundary — surveyed, in a defined CRS
2. DEM / topographic survey — 1 m or better preferred
3. Recent aerial or satellite imagery
4. Existing cadastral data and adjoining land parcels
5. Hydrology — rivers, wetlands, floodlines, boreholes
6. Geotechnical and soils data if available
7. Existing infrastructure — roads, power lines, water mains, servitudes
8. Environmental sensitivity, protected areas, heritage sites

**Regulatory and standards**
9. Botswana planning standards and development control regulations, in whatever form they exist
10. Applicable zoning schedules and land-use classifications
11. Road reserve widths and geometric standards
12. Facility provision standards — persons per school, clinic, police station — and required catchment distances
13. Minimum and typical plot sizes by category
14. The statutory approval pathway, and who the approving authority is

**Project**
15. Confirmed population target, density and housing mix assumptions
16. Development phasing horizon
17. A single empowered decision-maker on your side
18. The nominated Registered Town Planner

---

## 9. Voice and multilingual assistant

You asked for a voice agent using the BITC site as reference data, multilingual, with both call and web-screen interfaces. We are treating this as a **separate parallel workstream** — it shares almost no engineering with the GIS platform.

- **"Training on the site" is really retrieval, not training.** We crawl and index bitc.co.bw into a vector store; the assistant retrieves from it and cites sources. This is faster, cheaper, more accurate and — importantly — updatable when the site changes. Fine-tuning would be slower and would hallucinate more.
- **Languages:** English, French and Chinese as requested. **We strongly recommend adding Setswana** — for a Botswana-facing service it will matter more than French or Chinese. Note that Setswana speech recognition and synthesis quality is materially weaker than for the major languages; we will benchmark it honestly and show you real samples before committing to it.
- **Voice:** real-time speech pipeline with barge-in support; telephony via SIP/Twilio for the call channel; browser WebRTC for the web channel. The same assistant sits behind both.
- **Web screen:** chat plus voice widget, with live transcript and cited sources.
- **Timeline:** 4–6 weeks, in parallel with Sprint 0.
- **Guardrails:** scoped to BITC content, refuses out-of-scope questions, escalates to a human, full conversation logging.

---

## 10. Risks and how we manage them

| Risk | Impact | Mitigation |
|---|---|---|
| **4-week expectation vs. platform reality** | High | Sprint 0 explicitly scoped as a prototype, in writing, with an agreed exclusion list |
| **Site and standards data unavailable or poor** | **Highest** | Data request issued now (section 8); we survey what exists in week 0 and adjust scope before committing |
| Botswana standards not digitised or ambiguous | High | Registered planner embedded from week 1; standards encoded as reviewable rules, not buried in code |
| Plan not statutorily approvable | High | Approval pathway mapped in week 1; the planner signs, not the AI |
| AI produces plausible but wrong geometry | High | **LLM never generates geometry** — deterministic tools only; automated validation on every output |
| Scope creep | High | Fixed Sprint 0 scope; new requests go to a Phase 1 backlog, not into the sprint |
| GPL licensing contaminating commercial IP | Medium | Engine-first architecture keeps proprietary code outside GPL boundaries; legal review in Phase 1 |
| Esri licence cost or procurement delay | Medium | Engine is GIS-neutral; QGIS path available if Esri procurement stalls |
| Over-reliance on immature open source (PlanX / QCity) | Medium | Treated as reference implementations, never as critical-path dependencies |
| Connectivity limits in-country | Medium | Offline-capable desktop client; hybrid deployment |

---

## 11. How we make this succeed

1. **Fund the prototype, judge the prototype, then decide.** You proposed this sequencing; it is the right one.
2. **Get the data to us this week.** Nothing else in this document matters if week 1 has no site data.
3. **Name one decision-maker.** Committee-driven scope kills 4-week sprints.
4. **Embed the registered planner from day one**, not as a reviewer at the end.
5. **Weekly working demo.** Every Friday, live software — no slide decks.
6. **Accept the scope exclusions in writing.** They are what make 4 weeks possible.
7. **Position it honestly.** This is a force multiplier for professional planners, not a replacement for them. Sold that way, it will win approvals, funding and users. Sold as "AI plans the town", it will fail its first regulatory review.

---

## 12. Outstanding — comparable systems

You asked for examples of similar AI/GIS/urban planning systems we have developed. **This section must be completed from our actual project record before this document is sent.** We will not present projects we have not delivered; the client will verify, and the relationship will not survive it.

Where we lack a directly comparable urban-planning reference, the honest and effective position is to present adjacent evidence — geospatial engineering, optimisation, agentic AI systems — and then let Sprint 0 be the reference. That is precisely what a paid prototype is for, and it is a stronger argument than a case study.

---

## Proposed next steps

1. A 90-minute technical call — architecture walkthrough and Q&A with your planners and engineers
2. You issue the section 8 data pack
3. We confirm Sprint 0 scope, fixed fee and exclusion list in writing
4. Sprint 0 starts — demo account live at week 4

---

*All figures indicative and subject to discovery. Verified references: ArcGIS Urban API v1.0 (Esri Developer); PlanX suite and PlanX-UIP toolset (QGIS Plugin Repository / GitHub); QCity (North Road, July 2026, Australia-only pre-release).*
