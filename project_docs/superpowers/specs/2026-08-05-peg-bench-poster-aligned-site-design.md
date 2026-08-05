# PEG-Bench Poster-Aligned Research Site Design

**Status:** Approved direction A on 2026-08-05

## 1. Objective

Redesign the PEG-Bench GitHub Pages site as the mobile-first research companion to `poster_v2/poster-print-enhanced.pdf`. A visitor arriving from the poster QR code should immediately recognize the same project, understand the central prediction-to-execution argument without specialist knowledge, inspect the evidence like a reviewer, and reach the paper, code, and public artifact in one tap.

The site remains a dependency-light static publication under `docs/`. It is not a product landing page, trading dashboard, or conference microsite.

## 2. Public And Scientific Boundaries

- Do not display author names, affiliations, universities, conference names, workshop names, or venue-year labels in visible copy, metadata, structured data, alt text, filenames, or animation text.
- Do not expose licensed raw market data, reconstructable high-frequency features, full processed tensors, credentials, or private build artifacts.
- Do not imply tradable alpha, production readiness, clinical validity, or empirical validation outside the released market benchmark.
- Present AI coding, healthcare, cybersecurity, energy, and other domains as illustrative transfer scenarios only.
- Attach this exact disclosure to the application section: `Illustrative transfer scenarios. These domains were not evaluated in the released benchmark.`
- Preserve the released paper, evidence values, confidence intervals, source maps, and public artifact boundaries. The redesign changes presentation, not scientific results.

## 3. Audience And Entry Context

### Primary entry

A poster viewer scans a QR code on a phone. The visitor has 10-20 seconds to answer:

1. What is the research question?
2. What changed between prediction and execution?
3. Is the claim supported, and what are its limits?
4. Where are the paper and evidence?

### Secondary entry

A reviewer or researcher opens the GitHub Pages link on desktop and wants a structured path through the experimental contract, mechanisms, six-panel evidence, uncertainty, limitations, and reproducibility materials.

### Success criteria

- The first mobile viewport contains the project identity, thesis, one decisive result, and a paper or evidence action.
- The page remains comprehensible and navigable without JavaScript.
- No substantive table requires horizontal scrolling at 320-430 CSS pixels.
- Every empirical statement has a nearby evidence link or source label.
- Every application example is visually and verbally separated from benchmark evidence.

## 4. Design Direction

The visual system inherits the poster rather than introducing a separate web brand.

### Visual character

- Editorial-scientific, precise, restrained, and evidence-led.
- Asymmetric Swiss-style composition with strong left alignment and deliberate whitespace.
- Thin rules and alignment replace decorative cards and shadows.
- Flat evidence graphics replace stock imagery, abstract gradients, and ornamental illustrations.
- Square geometry is the default. Controls may use a 4px radius for focus and tactile clarity; content panels remain unrounded.

### Design dials

- Design variance: 6/10.
- Motion intensity: 4/10.
- Visual density: 5/10.

### Palette

- `--paper: #F8F7F2` for the poster-matched substrate.
- `--ink: #152536` for primary text and dark bands.
- `--teal: #0B6E69` for supported execution paths and evidence navigation.
- `--brick: #B8493C` for reversals, negative outcomes, cautions, and boundary labels.
- `--grey: #5E6B76` for secondary text.
- `--rule: #CDD9DF` for the single divider treatment.
- `--soft: #ECEEEA` for restrained evidence backgrounds.
- `--white: #FFFFFF` for accessible contrast where the paper tone is insufficient.

Teal and brick are semantic colors, not competing decoration. No gradients, glows, glass effects, or hue changes between sections.

### Typography

- Self-host Lato in the weights needed for body, headings, and numerical emphasis, matching the poster and avoiding a third-party font request.
- Use a system monospace stack only for protocol labels, compact values, and file paths.
- Desktop body: 17px with 1.65 line height; mobile body: 17px with 1.65 line height.
- Keep prose measures between 58ch and 68ch.
- Use discrete breakpoint sizes rather than viewport-width font scaling.
- Headlines are sentence case with zero letter spacing. Uppercase is reserved for short protocol labels.

## 5. Information Architecture

Keep the user's original eight-part reading order and preserve existing anchor intent:

1. `#problem`: Real-world problem.
2. `#concept`: Plain-language concept.
3. `#design`: Experiment design.
4. `#mechanism`: Mechanism interaction.
5. `#results`: Evidence results.
6. `#limits`: Limitations.
7. `#engineering`: Engineering application and transfer scenarios.
8. `#materials`: Paper, code, evidence, and reproduction materials.

The eight-part map appears once near the beginning. Section numbers remain useful wayfinding but do not repeat as oversized decorative badges.

## 6. Page Composition

### 6.1 Sticky navigation

Desktop navigation contains the PEG-Bench wordmark, `Problem`, `Design`, `Evidence`, `Limits`, and `Materials`, plus one primary `Paper` action. Mobile uses a compact sticky bar with the wordmark, a `Paper` action, and a native disclosure menu. The menu works without JavaScript; JavaScript may close it after an anchor selection.

All targets are at least 44 by 44 CSS pixels. Anchored sections account for the sticky-header offset.

### 6.2 Poster-continuation hero

The hero is the scan landing state and fits inside the first viewport on common desktop and 390x844 mobile screens while leaving a visible hint of the next section.

Content order:

1. `PEG-Bench` project marker.
2. H1: `When Prediction Leaderboards Can Fail`.
3. Supporting title: `Benchmarking the Prediction-Execution Gap`.
4. Thesis: `Prediction rankings can reverse after costs, turnover, and risk.`
5. Primary actions: `Read paper`, `View evidence`, and `GitHub`.
6. A poster-derived leader-identity trace and the result `4 of 6 active leader identities change at 3 bps`.

The hero contains no author strip, institution strip, venue label, generic “public technical report” eyebrow, or marketing superlatives.

### 6.3 Reading path

An unframed horizontal index follows the hero on desktop and becomes a compact two-column list on mobile. It presents the eight-part narrative without duplicating the main navigation.

### 6.4 Real-world problem and plain-language concept

Explain the distinction between a prediction, the action interface, and realized utility using one visual chain:

`Prediction score -> constrained action -> realized utility`

Pair a plain-language explanation with the formal evaluation question. The market setting is explicitly described as an evaluation test bed, not a deployable strategy claim.

### 6.5 Experiment design

Present the frozen contract through a poster-like ruled matrix: 90 days, two assets, two venues, time and volume views, horizons 5/20/50, chronological split, and 0/1/3/5/10 bps cost ladder. The R0-R4 ladder is visually primary and has a complete text equivalent.

At narrow widths, each matrix row becomes a stacked definition record with `Dimension`, `Frozen specification`, and `Why it matters` labels. It must not become a miniature desktop table.

### 6.6 Mechanism interaction

Use a single causal sequence rather than separate cards:

`signal -> action frequency -> position path -> costs and slippage -> risk and regime stress -> realized utility`

Each mechanism includes a plain-language effect and the corresponding report-card field. A small toy comparison between a high-turnover and a steadier model makes the mechanism understandable without adding a new empirical claim.

### 6.7 Evidence results

Lead with the poster's central ranking-change visual. The evidence hierarchy is:

1. LightGBM leads all six R0 panels.
2. Four of six clean-active R4 leader identities change at 3 bps.
3. No positive headline point estimate is statistically secure because its interval crosses zero.
4. The conclusion is a ranking-identity and evidence-boundary result, not a profit claim.

Desktop uses a wide ruled evidence table plus the poster-derived ranking trace. Mobile renders six vertical panel records with the panel, R0 leader and RankIC, R4 active leader and Sharpe, interval, and result label in that order. The same semantic table remains available to assistive technology.

The existing heatmap and signal-versus-utility figure stay below the central result as supporting evidence, with stable aspect ratios and concise text equivalents.

### 6.8 Limitations

Limitations appear immediately after evidence and use direct scientific language. Required boundaries include the 90-day window, six populated panels, licensed-data reconstruction boundary, limited method coverage, confidence-interval uncertainty, active-method eligibility rule, and the distinction between an evaluation result and a deployment claim.

Avoid defensive styling. Limitations use the same ruled system, with brick reserved for the most important caution.

### 6.9 Engineering applications

Use an unframed `Same gap, different stakes` matrix with four fields per row: `Prediction`, `Action`, `Realized utility`, and `Failure cost`.

Required scenarios:

- AI coding: code-generation score; accepted patch and tool calls; build, test, security, latency, and cost outcomes; failure cost from regressions or unsafe changes.
- Finance: return forecast; sizing, rebalance, and order path; fees, slippage, turnover, drawdown, and risk; failure cost from capital loss and instability.
- Healthcare: risk or treatment score; clinician or policy action; contraindications, workflow capacity, delay, and patient outcomes; failure cost from missed or harmful intervention.
- Cybersecurity: threat score; block, isolate, or escalate action; false-positive load, response latency, and service continuity; failure cost from breach or business interruption.
- Energy and industrial control: demand or fault forecast; scheduling or control action; stability, peak cost, reserve, and safety limits; failure cost from outage, damage, or unsafe operation.

The disclosure that these are transfer scenarios, not released empirical results, appears before the matrix and remains visible on mobile.

### 6.10 Materials

Group materials by reader task instead of file type:

- Read: paper and technical report.
- Inspect: result tables, source map, checksums, and evidence figures.
- Reproduce: evaluator, scripts, manifest, and public artifact instructions.
- Understand the boundary: licensed-data requirements and what the public snapshot cannot reconstruct.

Every link uses a valid relative Pages path or the renamed `computational-decision-lab/peg-bench` repository. External links open in a new tab with appropriate `rel` attributes; PDF links remain directly downloadable.

## 7. Motion System

Motion explains evidence and hierarchy. It never decorates empty space.

### Core animation

Create one transparent Lottie scene for the poster-derived R0-to-R4 leader trace:

- 30 fps, 2.4 seconds, no loop.
- First reveal six neutral R0 origins.
- Draw connectors with staged ease-out timing.
- Resolve unchanged paths in grey and changed paths in teal.
- Reveal the `4 / 6 changed` result last in brick.
- Do not animate numerical values, confidence intervals, or scientific labels into a misleading intermediate state.

The animation runs once when substantially visible. A static SVG with identical final-state information is present in the markup before the player loads and remains the fallback when JavaScript, the runtime, or motion is unavailable.

### Interaction motion

- Hero orchestration: 360-520ms opacity and translate reveal, once per load.
- Navigation and buttons: 120-180ms compositor-only feedback.
- Mobile disclosure: native state change with a short opacity/transform transition.
- `prefers-reduced-motion: reduce` disables orchestration and Lottie playback and displays the static final state.
- No infinite animation, scroll-jacking, parallax, blur animation, or layout-property animation.

## 8. Responsive Behavior

Validate at 320x568, 390x844, 430x932, 768x1024, 1280x800, and 1440x900.

- Mobile side padding is 18px at 320px and 22px at 390px and above.
- Hero actions remain one line per label and stack when necessary.
- Long words, paths, confidence intervals, and numerical labels wrap or use controlled overflow without clipping.
- Evidence graphics use stable aspect ratios and mobile-specific compositions rather than scaled-down desktop canvases.
- Tables transform into readable stacked records below 720px.
- Safe-area insets protect the sticky header and footer actions.
- Images below the fold lazy-load with explicit intrinsic dimensions.
- No horizontal page overflow at any required viewport.

## 9. Accessibility And Resilience

- Maintain semantic landmarks, a skip link, one H1, logical heading order, figure captions, table captions, row and column headers, and descriptive alt text.
- Meet WCAG 2.1 AA contrast, including focus indicators and small labels.
- Use `:focus-visible` with an offset that is visible on paper and dark surfaces.
- Do not use color alone to distinguish changed, unchanged, positive, negative, official, diagnostic, or target states.
- Keep all core content and links usable without JavaScript.
- If Lottie fails to load or parse, show the static SVG and do not display a broken-player message.
- Preserve readable zoom behavior through 200 percent.

## 10. Metadata And Sharing

- Use the poster-aligned paper title in the document title, Open Graph title, and ScholarlyArticle name.
- Update the description to state the prediction-to-execution thesis without venue or affiliation information.
- Create a poster-matched social preview containing only the project title, thesis, and evidence graphic.
- Keep the canonical URL on the renamed GitHub Pages origin.
- Include no person or organization schema.
- Update `dateModified` to the release date of the redesign.

## 11. Implementation Boundaries

Expected public-site changes are limited to:

- `docs/index.html` for structure and copy.
- `docs/styles.css` for the poster-aligned design system and responsive layouts.
- `docs/site.js` for progressive enhancement only.
- `docs/assets/` for self-hosted fonts, sanitized poster-derived SVGs, the Lottie JSON, a local player runtime and license if required, and the social preview.
- `PRODUCT.md` and `DESIGN.md` for the approved product and visual rules.
- A focused static-site validation script and tests for links, forbidden identity terms, semantic structure, animation fallback, and asset availability.

Do not introduce React, Vite, a component framework, analytics, forms, server code, or a build step for the webpage.

## 12. Verification And Release Gates

The redesign is complete only when all of the following pass:

1. Local static-site tests validate internal links, anchors, external repository URLs, required assets, metadata, and forbidden identity terms.
2. HTML validates without duplicate IDs or invalid landmark nesting.
3. Keyboard navigation, focus visibility, skip link, and mobile disclosure work in Chromium.
4. Reduced-motion mode shows the complete static evidence state.
5. The Lottie scene parses and renders at frame 0, midpoint, and final frame in the official Skia Skottie player before export.
6. Playwright screenshots at every required viewport show no overlap, clipping, blank evidence graphics, horizontal page overflow, or hidden actions.
7. Desktop and mobile screenshots visibly continue the poster's palette, typography, rule system, and evidence hierarchy.
8. The public repository contains no author names, affiliations, venue labels, credentials, licensed raw data, or private artifacts in the deployed `docs/` tree.
9. After push, GitHub Pages deployment succeeds and the live HTML, PDF, SVG, Lottie JSON, and image assets return the expected status and MIME type.
10. Live paper, GitHub, evidence, source-map, reproduction, and anchor links are exercised from the deployed page.

## 13. Accepted Trade-Offs

- Poster fidelity takes precedence over novelty, so the site deliberately uses the poster's warm paper, Lato typography, and restrained evidence palette.
- One small evidence animation is accepted as progressive enhancement; the research narrative and final evidence state never depend on it.
- The page remains detailed, but progressive disclosure and mobile-specific evidence records prevent the detail from becoming a miniature desktop paper.
- Transfer scenarios increase accessibility and engineering relevance, but their explicit disclosure prevents them from expanding the paper's empirical claims.
