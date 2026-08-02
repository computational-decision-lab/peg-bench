# PEG-Bench Research Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public-safe static research website and GitHub Pages deployment to the existing PEG-Bench repository.

**Architecture:** Keep the current `main` release artifact unchanged except for navigation metadata, then add a dependency-free static site under `docs/`. GitHub Actions packages `docs/` with the official Pages actions and deploys it from `main`; selected paper assets are copied into `docs/assets/` with no raw market data.

**Tech Stack:** Semantic HTML, CSS, progressive-enhancement JavaScript, GitHub Actions Pages actions, existing CSV/MD/PNG/PDF release artifacts.

## Global Constraints

- Preserve the public-safe boundary: do not add raw vendor L2 data or full processed benchmark tensors.
- Keep the site readable without JavaScript and responsive at mobile widths.
- Use the canonical URL `https://computational-decision-lab.github.io/peg-bench/` in metadata.
- Keep the paper and evidence claims aligned with the submitted 90D artifact.
- Do not introduce a build framework or external runtime dependency.

### Task 1: Curate public-safe site assets

**Files:**
- Create: `docs/assets/peg-bench-paper.pdf`
- Create: `docs/assets/peg-bench-thesis-overview.png`
- Create: `docs/assets/winner-shift.png`
- Create: `docs/assets/r4-sharpe-heatmap.png`
- Create: `docs/assets/rankic-vs-r4-sharpe.png`

- [x] Copy the public technical report and existing PNG figures from the local PEG-Bench package into the clean clone.
- [ ] Rasterize the two selected result PDFs for browser-safe figure rendering; keep the source result files in the artifact directories unchanged.
- [ ] Verify that all five assets exist, have non-zero sizes, and contain no raw-data paths in their names or metadata.

### Task 2: Build the static research narrative

**Files:**
- Create: `docs/index.html`
- Create: `docs/styles.css`
- Create: `docs/site.js`

**Interfaces:**
- `docs/index.html` references `styles.css`, `site.js`, and the five files under `assets/`.
- The evidence toggle uses buttons with `data-view="prediction"` and `data-view="execution"` plus `[data-panel]` rows.

- [ ] Add semantic sections for the thesis hero, evidence snapshot, R0-to-R4 ladder, six-panel results, paper/artifact links, and release boundary.
- [ ] Add deterministic SEO and social metadata, JSON-LD scholarly article data, skip navigation, descriptive figure alt text, and visible focus states.
- [ ] Implement the responsive research-instrument visual system in `styles.css` with no gradients, no oversized card grid, and reduced-motion handling.
- [ ] Implement the evidence toggle and progressive enhancement in `site.js`; preserve the default prediction view when JavaScript is unavailable.

### Task 3: Publish through GitHub Pages

**Files:**
- Create: `.github/workflows/pages.yml`
- Modify: `README.md`
- Modify: `RELEASE_SCOPE.md`

- [ ] Add a workflow triggered by pushes to `main` and manual dispatch that uploads `docs/` and deploys it with `actions/deploy-pages`.
- [ ] Update the README with the website URL, paper/artifact navigation, and the same data-license boundary used on the site.
- [ ] Add a short release note stating that the website is a presentation layer over the existing public-safe artifact.
- [x] Commit the curated site and workflow on the clone's `main` branch, then push to `computational-decision-lab/peg-bench`.
- [ ] Configure repository metadata and Pages workflow mode with the authenticated GitHub CLI.

### Task 4: Verify the release

**Files:**
- Verify: all files above and GitHub Actions/Pages state.

- [ ] Run `git diff --check`, HTML path checks, asset size checks, and a local `python3 -m http.server` smoke test.
- [ ] Inspect the pushed commit and workflow run, then query the Pages API.
- [ ] Fetch the live canonical URL and verify HTTP 200 plus the paper, GitHub, and artifact links.
- [ ] Report any remaining GitHub Pages propagation delay explicitly instead of treating it as a successful deployment.
