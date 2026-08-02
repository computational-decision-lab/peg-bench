# PEG-Bench Research Site Design

## Goal

Publish a public, static research site for PEG-Bench at `https://colin-wang-research.github.io/peg-bench/` while preserving the repository's existing public-safe artifact boundary.

## Audience and narrative

The primary audience is a researcher or reviewer arriving from GitHub or a paper citation. The first viewport must answer three questions: what fails, how PEG-Bench measures the failure, and what the completed evidence says. The narrative is:

1. Offline prediction quality is an incomplete deployment proxy.
2. PEG-Bench forces each method through a fixed R0-to-R4 realism ladder.
3. Across six completed panels, the clean active R4 winner changes in four panels; in the two unchanged panels, the active winner remains negative after costs.

The site must distinguish official clean-active leaderboard evidence from diagnostic wrapper/action-interface rows and must not claim production execution, tradable alpha, or cross-market validation.

## Site structure

- `docs/index.html`: one-page research narrative with semantic sections for thesis, ladder, results, paper, artifact, and release boundary.
- `docs/styles.css`: responsive visual system based on the repository's research-instrument direction; no framework or external runtime dependency.
- `docs/site.js`: progressive enhancement only for the current year; content remains readable without JavaScript.
- `docs/assets/`: intentionally selected paper PDF and derived figures only.
- `.github/workflows/pages.yml`: GitHub Pages deployment using the official Pages actions.

## Content and assets

The site includes the public technical report and figures derived from the public-safe local research package. It links back to the existing repository artifact rather than duplicating raw data. The public page repeats the licensing disclaimer: raw vendor L2 feeds and full processed tensors are not redistributed; full raw-data reconstruction requires authorized provider access.

## Metadata and accessibility

The page has a deterministic title, description, canonical URL, Open Graph/Twitter metadata, theme color, favicon, JSON-LD ScholarlyArticle data, `lang="en"`, skip link, semantic headings, keyboard focus states, reduced-motion handling, and descriptive alt text. No JavaScript is required for core navigation or reading.

## Deployment and verification

GitHub Actions uploads `docs/` as a Pages artifact and deploys it from `main`. Verification includes HTML/link/path checks, `git diff --check`, asset hash checks, workflow YAML inspection, a local HTTP smoke test, and a live HTTP check against the GitHub Pages URL after deployment.

## Out of scope

No client-side data explorer, no raw-data download, no external analytics, no framework migration, and no claim expansion beyond the submitted evidence.
