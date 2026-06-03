# Resources Index — local corpus navigation manifest

> **Generated** by `scripts/tools/build_resources_index.py`. Do not edit by hand.
> Rebuild: `python scripts/tools/build_resources_index.py`.

**Grounding rule (CLAUDE.md § Local Academic / Project-Source Grounding):**
before citing training memory on any topic below, ground in the local source.
Where a **curated extract** exists in `docs/institutional/literature/`, that
extract is the **canonical citation source** (page-cited) — fetch it via the
`research-catalog` MCP (`get_literature_excerpt`), do NOT paraphrase the raw PDF
from memory. If no local source supports a claim, say **UNSUPPORTED**.

**6 resources indexed.**

| Resource | Topic | Curated extract (canonical cite) | Key terms (first page) |
|---|---|---|---|
| `Advances_in_Financial_Machine_Learning_Lopez_de_Prado_2018.pdf` | Advances in Financial Machine Learning — full book (Lopez de Prado 2018) | `docs/institutional/literature/lopez_de_prado_2018_afml_ch_3_7_8.md` | Praise for Advances in Financial Machine Learning In his new book Advances in Financial Machine Learning, noted financial scholar Marcos L´o |
| `Howard_2026_Value_Area_Breakouts_ES.pdf` | Value-area breakout / stop methodology (Howard 2026, UNREVIEWED preprint) | `docs/institutional/literature/howard_2026_value_area_breakouts_es.md` | Stop Distance, Exit Methodology, and Signal Preservation in Intraday Value Area Breakouts: Evidence from E-mini S&P 500 Futures Theo Johann  |
| `ml4am_code_companion/` | ML4AM code companion (Lopez de Prado notebooks) | `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` | (subdir corpus — see its README / page files) |
| `projectx_api_spec_2026_05_16.md` | Broker — ProjectX/TopstepX API spec (live execution) | — | (markdown — read directly) |
| `prop-firm-official-rules.md` | Prop-firm official rules (account death / MLL) | — | (markdown — read directly) |
| `Tolusic_2026_AMT_Inventory_Dynamics.pdf` | Auction-market-theory inventory dynamics (Tolusic 2026, UNREVIEWED preprint) | `docs/institutional/literature/tolusic_2026_amt_inventory_dynamics.md` | Auction Market Theory as Emergent Property Auction Market Theory as an Emergent Property of Inventory Dynamics: The First Formal Mathematica |

## How to ground cheaply

1. Find the row for your topic above.
2. If it has a **curated extract**, cite from that file (page numbers included).
3. If not (`—`), extract the TOC + 3 mid-document pages from the PDF before
   characterizing or dismissing it (terminology differs across sources — see
   institutional-rigor.md § 7 extract-before-dismiss rule).
4. Never cite a resource you have not opened this session as if you read it.

