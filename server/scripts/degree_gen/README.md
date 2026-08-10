# Degree-requirement data generation (offline)

These scripts rebuild the `data/degree_requirements/` and `data/minors/` corpus from the
public Sabancı degree-detail pages. This is an **offline / one-time** pipeline — the running
app never calls it. Runtime ingestion is `server/scripts/ingest_degree_requirements.py`.

## Majors (11 programs x 4 admit terms)

Program codes: CS=`BSCS`, IE=`BSMS`, BIO=`BSBIO`, ME=`BSME`, EE=`BSEE`, PSY=`BAPSY`,
ECON=`BAECON`, DSA=`BSDSA`, MAT=`BSMAT`, MAN=`BAMAN`, PSIR=`BAPSIR`.
Terms: `202201 202301 202401 202501`.

MAN and PSIR are rebuilt directly from the live official degree pages:

```bash
python scrape_official_degree_reqs.py configs/config_MAN.json ../../../data/degree_requirements/MAN
python scrape_official_degree_reqs.py configs/config_PSIR.json ../../../data/degree_requirements/PSIR
```

The scraper keeps the official program identifiers (`BAMAN`, `BAPSIR`) while writing the
repository aliases (`MAN`, `PSIR`). PSIR's Political Science and International Relations core
elective pools remain separate because both carry independent 12-SU minimums. Official summary
inconsistencies are recorded as source discrepancies; the generator does not invent courses or
equivalencies to make totals agree.

1. **Download pool pages** (Core/Area/Free electives + FENS/FASS/SBS faculty pools) into
   `pages/<P_PROGRAM>/{CEL,AEL,FEL,FCFENS,FCFASS,FCSBS}_<term>.html` with `curl` against
   `SU_DEGREE.p_list_courses?...&P_AREA=<PROG>_CEL|AEL|FEL | FC_FENS|FC_FASS|FC_SOM`.
   Note pool-code quirks: **EE** uses `BSEE_CEL/ARE/FRE`; **PSY** uses `BAPSY_COR/ARE/FRE`.
2. `python parse_pools.py <pages_dir> <parsed_pools.json>` — raw HTML → catalog + pools.
3. `python gen_degree_reqs.py configs/config_<PROG>.json <parsed_pools.json> data/degree_requirements/<PROG>`
   (CS was built with the older `gen_cs_degree_reqs.py`; identical output schema.)

Each `config_<PROG>.json` carries program metadata, University/Required course lists per
term, and per-term flags: `math_choice`, `math_212_or_201_202`, `prog_choice`, `choices`,
`math_exclusion_2025`, `pool_suffix`, `hum_min_courses`, `extra_pools`, `faculty_rule`,
`eng_ects`/`bsci_ects` (null to omit Engineering/Basic-Science for science/BA programs).

## Minors (17 minors × 4 terms)

`build_minors.py <minor_pages_dir> data/minors <external_pool_dir>` — minors are enumerated
inline on the degree page; 3 (ENTREP/MKTG/DECB) have external area-elective pools fetched
into `<external_pool_dir>` as `<PROGRAM>__<AREA>__<term>.html`.

## After (re)generating data

```
npm run registry            # rebuild data/curricula/registry.jsonl
npm run ingest:degrees:reset  # re-embed into ChromaDB (server/chroma_store)
```
