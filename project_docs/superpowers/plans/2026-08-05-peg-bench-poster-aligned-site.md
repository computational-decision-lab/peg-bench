# PEG-Bench Poster-Aligned Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the PEG-Bench GitHub Pages site as a poster-matched, reviewer-auditable, mobile-first research reader with high-value AI transfer scenarios and one restrained evidence animation.

**Architecture:** Keep the existing one-page, dependency-light static site under `docs/`. Use semantic HTML as the complete source of truth, CSS for poster-aligned responsive compositions, and JavaScript only for progressive enhancement of the mobile disclosure and a local Lottie animation; every animated claim has a static SVG and text fallback.

**Tech Stack:** HTML5, CSS, vanilla JavaScript, Node.js 22 built-in test runner, Lottie JSON, vendored `lottie-web` light runtime, Playwright CLI, GitHub Pages Actions.

## Global Constraints

- Preserve the section order `#problem`, `#concept`, `#design`, `#mechanism`, `#results`, `#limits`, `#engineering`, `#materials`.
- Do not display paper author names, affiliations, universities, conference names, workshop names, or venue-year labels in deployed HTML, metadata, alt text, SVG text, or animation text.
- Use `#F8F7F2`, `#152536`, `#0B6E69`, `#B8493C`, `#5E6B76`, `#CDD9DF`, `#ECEEEA`, and `#FFFFFF` with no gradients, glows, glass effects, or decorative shadows.
- Keep the site usable without JavaScript and meet WCAG 2.1 AA.
- Label the cross-domain section exactly: `Illustrative transfer scenarios. These domains were not evaluated in the released benchmark.`
- Do not change released result values, confidence intervals, paper PDFs, artifact tables, or public-data boundaries.
- Do not add React, Vite, a component framework, analytics, forms, server code, or a webpage build step.
- The Lottie animation is 960x360, 30 fps, frames 0-71 (`op: 72`), non-looping, transparent, and replaceable by an equivalent static SVG.
- Validate 320x568, 390x844, 430x932, 768x1024, 1280x800, and 1440x900.
- Work only in the isolated public clone, never the dirty research checkout.

---

### Task 1: Align The Product And Design Contracts

**Files:**
- Modify: `PRODUCT.md`
- Modify: `DESIGN.md`
- Create: `site_tests/site-contract.test.mjs`

**Interfaces:**
- Consumes: approved design spec at `project_docs/superpowers/specs/2026-08-05-peg-bench-poster-aligned-site-design.md`.
- Produces: canonical product copy, design tokens, viewports, and a reusable Node test helper for later tasks.

- [ ] **Step 1: Write the contract test before changing the design documents**

Create `site_tests/site-contract.test.mjs` with the shared helpers and first test:

```js
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const read = (path) => readFileSync(resolve(root, path), "utf8");
const exists = (path) => existsSync(resolve(root, path));

test("product and design contracts encode the approved direction", () => {
  const product = read("PRODUCT.md");
  const design = read("DESIGN.md");
  assert.match(product, /poster QR/i);
  assert.match(product, /Illustrative transfer scenarios\. These domains were not evaluated in the released benchmark\./);
  for (const color of ["#F8F7F2", "#152536", "#0B6E69", "#B8493C", "#5E6B76", "#CDD9DF", "#ECEEEA"]) {
    assert.ok(design.includes(color), `missing poster color ${color}`);
  }
  assert.match(design, /320x568/);
  assert.match(design, /1440x900/);
  assert.match(design, /Lato/);
  assert.match(design, /static SVG fallback/i);
});

export { exists, read, root };
```

- [ ] **Step 2: Run the test and verify it fails against the legacy contracts**

Run: `node --test site_tests/site-contract.test.mjs`

Expected: FAIL because the existing `PRODUCT.md` and `DESIGN.md` do not contain the poster QR entry, exact transfer disclosure, complete poster palette, required viewports, or static animation fallback.

- [ ] **Step 3: Replace the legacy product and design decisions with the approved contracts**

Update `PRODUCT.md` so it states:

```markdown
## Primary Entry

A visitor scans the poster QR code on a phone and needs the thesis, decisive result, evidence boundary, paper, and GitHub route within the first reading sequence.

## Transfer Disclosure

Illustrative transfer scenarios. These domains were not evaluated in the released benchmark.
```

Keep the existing researcher, reviewer, practitioner, public-safe artifact, and anti-dashboard requirements. Update `DESIGN.md` with the exact global palette, Lato typography, square geometry, one-hairline rule system, eight-section order, required viewports, mobile record transformation, Lottie timing, and static SVG fallback from the approved spec.

- [ ] **Step 4: Run the contract test and repository whitespace check**

Run: `node --test site_tests/site-contract.test.mjs && git diff --check`

Expected: 1 test passes; `git diff --check` prints no errors.

- [ ] **Step 5: Commit the contract update**

```bash
git add PRODUCT.md DESIGN.md site_tests/site-contract.test.mjs
git commit -m "docs: align site contracts with poster"
```

---

### Task 2: Curate Poster-Matched Public Assets

**Files:**
- Create: `docs/assets/peg-evaluation-chain.svg`
- Create: `docs/assets/peg-ranking-shift.svg`
- Create: `docs/assets/fonts/Lato-Regular.ttf`
- Create: `docs/assets/fonts/Lato-Bold.ttf`
- Create: `docs/assets/fonts/Lato-Heavy.ttf`
- Create: `docs/assets/fonts/OFL.txt`
- Modify: `site_tests/site-contract.test.mjs`

**Interfaces:**
- Consumes: anonymous figure SVGs from `/opt/projects/research/peg/kdd-mlf-2026/poster_v2/assets/` and system Lato files from `/usr/share/fonts/truetype/lato/`.
- Produces: stable, same-origin assets referenced by HTML, CSS, social metadata, and the motion fallback.

- [ ] **Step 1: Add a failing public-asset and identity-safety test**

Append:

```js
test("poster assets and self-hosted fonts are present and identity safe", () => {
  const required = [
    "docs/assets/peg-evaluation-chain.svg",
    "docs/assets/peg-ranking-shift.svg",
    "docs/assets/fonts/Lato-Regular.ttf",
    "docs/assets/fonts/Lato-Bold.ttf",
    "docs/assets/fonts/Lato-Heavy.ttf",
    "docs/assets/fonts/OFL.txt"
  ];
  required.forEach((path) => assert.ok(exists(path), `missing ${path}`));
  const identityPattern = /Ke Wang|Yangyang Liang|Xiaorui Tang|Georgia Tech|Guangdong University|Hong Kong University|HKUST|KDD|MLF 2026|NeurIPS/i;
  for (const path of ["docs/assets/peg-evaluation-chain.svg", "docs/assets/peg-ranking-shift.svg"]) {
    assert.doesNotMatch(read(path), identityPattern, `${path} contains excluded identity text`);
  }
});
```

- [ ] **Step 2: Run only the new test and verify the missing-asset failure**

Run: `node --test --test-name-pattern="poster assets" site_tests/site-contract.test.mjs`

Expected: FAIL with `missing docs/assets/peg-evaluation-chain.svg`.

- [ ] **Step 3: Copy the verified anonymous assets and fonts**

```bash
mkdir -p docs/assets/fonts
cp /opt/projects/research/peg/kdd-mlf-2026/poster_v2/assets/editorial_hero_final.svg docs/assets/peg-evaluation-chain.svg
cp /opt/projects/research/peg/kdd-mlf-2026/poster_v2/assets/editorial_ranking_print_enhanced.svg docs/assets/peg-ranking-shift.svg
cp /usr/share/fonts/truetype/lato/Lato-Regular.ttf docs/assets/fonts/Lato-Regular.ttf
cp /usr/share/fonts/truetype/lato/Lato-Bold.ttf docs/assets/fonts/Lato-Bold.ttf
cp /usr/share/fonts/truetype/lato/Lato-Heavy.ttf docs/assets/fonts/Lato-Heavy.ttf
cp /usr/share/doc/fonts-lato/copyright docs/assets/fonts/OFL.txt
```

Confirm that both SVGs have view boxes, accessible roles, poster colors, no scripts, no external URLs, and no excluded identity text.

- [ ] **Step 4: Run asset tests and record hashes**

Run:

```bash
node --test --test-name-pattern="poster assets" site_tests/site-contract.test.mjs
sha256sum docs/assets/peg-evaluation-chain.svg docs/assets/peg-ranking-shift.svg docs/assets/fonts/Lato-*.ttf
git diff --check
```

Expected: the asset test passes, six hashes print, and the whitespace check is clean.

- [ ] **Step 5: Commit the curated assets**

```bash
git add docs/assets/peg-evaluation-chain.svg docs/assets/peg-ranking-shift.svg docs/assets/fonts site_tests/site-contract.test.mjs
git commit -m "feat: add poster-matched public assets"
```

---

### Task 3: Rebuild The Semantic Research Narrative

**Files:**
- Modify: `docs/index.html`
- Modify: `site_tests/site-contract.test.mjs`

**Interfaces:**
- Consumes: existing verified result values and public links; poster SVG assets from Task 2.
- Produces: the complete no-JavaScript reading experience, stable section anchors, mobile disclosure markup, responsive data labels, and application transfer matrix.

- [ ] **Step 1: Add failing semantic, content, privacy, anchor, and asset-reference tests**

Append these helpers and tests:

```js
const attributes = (html, name) => [...html.matchAll(new RegExp(`${name}=["']([^"']+)["']`, "g"))].map((match) => match[1]);

test("page follows the approved eight-part narrative", () => {
  const html = read("docs/index.html");
  const ids = ["problem", "concept", "design", "mechanism", "results", "limits", "engineering", "materials"];
  let cursor = -1;
  for (const id of ids) {
    const next = html.indexOf(`id="${id}"`);
    assert.ok(next > cursor, `${id} is missing or out of order`);
    cursor = next;
  }
  assert.match(html, /When Prediction Leaderboards Can Fail/);
  assert.match(html, /Prediction rankings can reverse after costs, turnover, and risk\./);
  assert.match(html, /4 of 6 active leader identities change at 3 bps/);
});

test("application scenarios are useful but never presented as evidence", () => {
  const html = read("docs/index.html");
  assert.match(html, /Illustrative transfer scenarios\. These domains were not evaluated in the released benchmark\./);
  for (const domain of ["AI coding", "Finance", "Healthcare", "Cybersecurity", "Energy and industrial control"]) {
    assert.ok(html.includes(domain), `missing ${domain}`);
  }
  for (const field of ["Prediction", "Action", "Realized utility", "Failure cost"]) {
    assert.ok(html.includes(field), `missing application field ${field}`);
  }
});

test("deployed page omits excluded identity and venue text", () => {
  const html = read("docs/index.html");
  const excluded = /Ke Wang|Yangyang Liang|Xiaorui Tang|Georgia Tech|Guangdong University|Hong Kong University|HKUST|KDD|MLF 2026|NeurIPS/i;
  assert.doesNotMatch(html, excluded);
  assert.doesNotMatch(html, /Public technical report \/ research benchmark/i);
});

test("all local page references resolve and all fragment links have targets", () => {
  const html = read("docs/index.html");
  const ids = new Set(attributes(html, "id"));
  for (const href of attributes(html, "href")) {
    if (href.startsWith("#")) assert.ok(ids.has(href.slice(1)), `missing anchor ${href}`);
    if (href.startsWith("./")) assert.ok(exists(`docs/${href.slice(2)}`), `missing local href ${href}`);
  }
  for (const src of attributes(html, "src")) {
    if (src.startsWith("./")) assert.ok(exists(`docs/${src.slice(2)}`), `missing local src ${src}`);
  }
});
```

- [ ] **Step 2: Run the new tests and verify the legacy page fails**

Run: `node --test --test-name-pattern="approved eight|application scenarios|excluded identity|local page references" site_tests/site-contract.test.mjs`

Expected: FAIL on the poster title, application scenarios, and missing new asset references.

- [ ] **Step 3: Replace the header and hero with the poster-continuation structure**

Use one H1 and this interface contract:

```html
<header class="site-header">
  <nav class="nav shell" aria-label="Primary navigation">
    <a class="wordmark" href="#top" aria-label="PEG-Bench overview">PEG-Bench</a>
    <div class="nav-primary" aria-label="Research sections">
      <a href="#problem">Problem</a>
      <a href="#design">Design</a>
      <a href="#results">Evidence</a>
      <a href="#limits">Limits</a>
      <a href="#materials">Materials</a>
    </div>
    <a class="nav-paper" href="./assets/peg-bench-technical-report.pdf">Paper</a>
    <details class="mobile-menu">
      <summary aria-label="Open navigation"><span aria-hidden="true"></span></summary>
      <nav aria-label="Mobile navigation">
        <a href="#problem">Problem</a>
        <a href="#concept">Concept</a>
        <a href="#design">Design</a>
        <a href="#mechanism">Mechanism</a>
        <a href="#results">Evidence</a>
        <a href="#limits">Limits</a>
        <a href="#engineering">Applications</a>
        <a href="#materials">Materials</a>
      </nav>
    </details>
  </nav>
</header>
<section id="top" class="hero band">
  <div class="shell hero-grid">
    <div class="hero-copy">
      <p class="project-marker">PEG-Bench</p>
      <h1>When Prediction Leaderboards Can Fail</h1>
      <p class="hero-subtitle">Benchmarking the Prediction-Execution Gap</p>
      <p class="hero-thesis">Prediction rankings can reverse after costs, turnover, and risk.</p>
      <div class="hero-actions">
        <a class="button button-primary" href="./assets/peg-bench-technical-report.pdf">Read paper</a>
        <a class="button button-secondary" href="#results">View evidence</a>
        <a class="text-link" href="https://github.com/computational-decision-lab/peg-bench" target="_blank" rel="noreferrer">GitHub</a>
      </div>
    </div>
    <figure class="hero-evidence">
      <img src="./assets/peg-evaluation-chain.svg" width="2400" height="3200" alt="Evaluation chain from prediction ranking through a shared action interface and friction accounting to post-cost active selection">
      <figcaption><strong>4 of 6</strong><span>active leader identities change at 3 bps</span></figcaption>
    </figure>
  </div>
</section>
```

- [ ] **Step 4: Recompose all eight sections without changing evidence values**

Retain the accurate existing explanations and table values while changing their hierarchy:

- Use one eight-part reader map after the hero.
- Use `Prediction score -> constrained action -> realized utility` in `#problem` and `#concept`.
- Preserve the frozen contract and R0-R4 ladder in `#design`.
- Present the causal sequence in `#mechanism`.
- Put `peg-ranking-shift.svg` before the six-panel result table in `#results`.
- Add `data-label` to every table data cell so CSS can create mobile records.
- Keep uncertainty and protocol caveats adjacent to the result table.
- Keep limitations immediately after results.
- Add the exact transfer disclosure and five application rows under `#engineering`.
- Group `#materials` links under Read, Inspect, Reproduce, and Boundary.

Use this row contract for every application:

```html
<article class="application-row">
  <h3>AI coding</h3>
  <dl>
    <div><dt>Prediction</dt><dd>Code-generation or patch-quality score</dd></div>
    <div><dt>Action</dt><dd>Accepted patch and tool calls</dd></div>
    <div><dt>Realized utility</dt><dd>Build, test, security, latency, and cost outcomes</dd></div>
    <div><dt>Failure cost</dt><dd>Regression, unsafe change, or wasted review time</dd></div>
  </dl>
</article>
```

Repeat the same exact `dl` structure for Finance, Healthcare, Cybersecurity, and Energy and industrial control using the approved spec copy.

- [ ] **Step 5: Run semantic tests and an HTML structure smoke check**

Run:

```bash
node --test site_tests/site-contract.test.mjs
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path
class Parser(HTMLParser):
    pass
Parser().feed(Path("docs/index.html").read_text())
print("HTML parse: OK")
PY
git diff --check
```

Expected: all contract tests pass, `HTML parse: OK` prints, and no whitespace errors appear.

- [ ] **Step 6: Commit the semantic rebuild**

```bash
git add docs/index.html site_tests/site-contract.test.mjs
git commit -m "feat: rebuild the poster-aligned research narrative"
```

---

### Task 4: Implement The Poster-Aligned Responsive Design System

**Files:**
- Modify: `docs/styles.css`
- Modify: `site_tests/site-contract.test.mjs`

**Interfaces:**
- Consumes: semantic classes and `data-label` attributes from Task 3; Lato files from Task 2.
- Produces: desktop editorial layout, mobile navigation and records, stable chart dimensions, accessible states, and reduced-motion CSS used by Task 5.

- [ ] **Step 1: Add a failing CSS contract test**

Append:

```js
test("CSS implements the poster system and mobile reading contract", () => {
  const css = read("docs/styles.css");
  for (const token of ["#F8F7F2", "#152536", "#0B6E69", "#B8493C", "#5E6B76", "#CDD9DF", "#ECEEEA"]) {
    assert.ok(css.includes(token), `missing CSS color ${token}`);
  }
  assert.match(css, /@font-face[\s\S]*Lato-Regular\.ttf/);
  assert.match(css, /@media \(max-width: 720px\)/);
  assert.match(css, /td::before[\s\S]*attr\(data-label\)/);
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)/);
  assert.match(css, /min-height:\s*44px/);
  assert.doesNotMatch(css, /linear-gradient|radial-gradient|backdrop-filter/i);
});
```

- [ ] **Step 2: Run the CSS test and verify it fails against the old palette and mobile table behavior**

Run: `node --test --test-name-pattern="CSS implements" site_tests/site-contract.test.mjs`

Expected: FAIL on exact palette, Lato font loading, 720px mobile records, and 44px targets.

- [ ] **Step 3: Replace global tokens, type, spacing, and interaction states**

Start `docs/styles.css` with this contract:

```css
@font-face { font-family: "Lato"; src: url("./assets/fonts/Lato-Regular.ttf") format("truetype"); font-style: normal; font-weight: 400; font-display: swap; }
@font-face { font-family: "Lato"; src: url("./assets/fonts/Lato-Bold.ttf") format("truetype"); font-style: normal; font-weight: 700; font-display: swap; }
@font-face { font-family: "Lato"; src: url("./assets/fonts/Lato-Heavy.ttf") format("truetype"); font-style: normal; font-weight: 900; font-display: swap; }

:root {
  --paper: #F8F7F2;
  --ink: #152536;
  --teal: #0B6E69;
  --brick: #B8493C;
  --grey: #5E6B76;
  --rule: #CDD9DF;
  --soft: #ECEEEA;
  --white: #FFFFFF;
  --sans: "Lato", "Avenir Next", "Segoe UI", sans-serif;
  --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  --header-height: 68px;
}

*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: calc(var(--header-height) + 18px); }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 17px; line-height: 1.65; letter-spacing: 0; }
a, button, summary { min-height: 44px; }
:focus-visible { outline: 3px solid var(--brick); outline-offset: 4px; }
.shell { width: min(calc(100% - 48px), 1180px); margin-inline: auto; }
```

Use one 1px `--rule` divider treatment, no content-card shadow, 0px content radius, and 4px control radius. Use discrete media-query heading sizes rather than `vw`-based type.

- [ ] **Step 4: Implement desktop composition and mobile-specific evidence records**

Desktop uses a 12-column grid, an asymmetric hero, ruled result and application rows, and a maximum prose measure of 68ch. At `max-width: 720px`, implement the record transformation:

```css
@media (max-width: 720px) {
  .shell { width: min(calc(100% - 36px), 1180px); }
  .nav-primary { display: none; }
  .mobile-menu { display: block; }
  .hero-grid { grid-template-columns: 1fr; }
  .hero h1 { font-size: 40px; line-height: 1.06; }
  .reader-map ol { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .responsive-table thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
  .responsive-table, .responsive-table tbody, .responsive-table tr, .responsive-table th, .responsive-table td { display: block; width: 100%; }
  .responsive-table tr { padding-block: 18px; border-top: 1px solid var(--rule); }
  .responsive-table td { display: grid; grid-template-columns: minmax(108px, 0.42fr) 1fr; gap: 14px; padding: 4px 0; border: 0; }
  .responsive-table td::before { content: attr(data-label); color: var(--grey); font-family: var(--mono); font-size: 11px; font-weight: 700; text-transform: uppercase; }
  .application-row dl { grid-template-columns: 1fr; }
  .figure-grid { grid-template-columns: 1fr; }
}

@media (max-width: 359px) {
  .shell { width: min(calc(100% - 36px), 1180px); }
  .hero h1 { font-size: 36px; }
  .reader-map ol { grid-template-columns: 1fr; }
}
```

Add safe-area padding, explicit image aspect ratios, `overflow-wrap: anywhere` for paths only, and `scroll-margin-top` for all anchored sections. Do not use global horizontal overflow as a table solution.

- [ ] **Step 5: Run CSS tests and local screenshots at one mobile and one desktop viewport**

Run:

```bash
node --test site_tests/site-contract.test.mjs
python3 -m http.server 4173 --directory docs >/tmp/peg-site-server.log 2>&1 &
PEG_SITE_PID=$!
npx playwright screenshot --full-page --viewport-size=390,844 http://127.0.0.1:4173/ /tmp/peg-site-390.png
npx playwright screenshot --full-page --viewport-size=1440,900 http://127.0.0.1:4173/ /tmp/peg-site-1440.png
kill "$PEG_SITE_PID"
git diff --check
```

Expected: tests pass; both screenshots are nonblank; mobile has a visible navigation control, no miniature horizontal table, and no horizontal page overflow; desktop visibly matches the poster palette and rule system.

- [ ] **Step 6: Commit the responsive design system**

```bash
git add docs/styles.css site_tests/site-contract.test.mjs
git commit -m "style: add poster-aligned responsive design system"
```

---

### Task 5: Add The Evidence Lottie As Progressive Enhancement

**Files:**
- Create: `docs/assets/peg-ranking-trace.json`
- Create: `docs/assets/vendor/lottie-light.min.js`
- Create: `docs/assets/vendor/lottie-web-LICENSE.md`
- Modify: `docs/index.html`
- Modify: `docs/site.js`
- Modify: `docs/styles.css`
- Modify: `site_tests/site-contract.test.mjs`

**Interfaces:**
- Consumes: `peg-ranking-shift.svg`, `.motion-fallback`, `#ranking-motion`, and poster colors.
- Produces: one non-looping 72-frame technical trace with a static final-state fallback and an enhancement controller that is inert under reduced motion.

- [ ] **Step 1: Add failing animation structure and fallback tests**

Append:

```js
test("ranking animation has the approved timing and a static fallback", () => {
  const html = read("docs/index.html");
  const js = read("docs/site.js");
  assert.ok(exists("docs/assets/peg-ranking-trace.json"));
  assert.ok(exists("docs/assets/vendor/lottie-light.min.js"));
  const motion = JSON.parse(read("docs/assets/peg-ranking-trace.json"));
  assert.deepEqual({ fr: motion.fr, ip: motion.ip, op: motion.op, w: motion.w, h: motion.h }, { fr: 30, ip: 0, op: 72, w: 960, h: 360 });
  assert.ok(motion.layers.length >= 12, "expected paths and endpoints for six evidence rows");
  assert.match(html, /id="ranking-motion"/);
  assert.match(html, /class="motion-fallback"[\s\S]*peg-ranking-shift\.svg/);
  assert.match(html, /data-lottie-src="\.\/assets\/peg-ranking-trace\.json"/);
  assert.match(js, /prefers-reduced-motion/);
  assert.match(js, /loop:\s*false/);
  assert.match(js, /data_failed/);
});
```

- [ ] **Step 2: Run the animation test and verify the missing-scene failure**

Run: `node --test --test-name-pattern="ranking animation" site_tests/site-contract.test.mjs`

Expected: FAIL because the Lottie JSON, runtime, markup, and controller do not exist.

- [ ] **Step 3: Set up the official Skia Skottie player in a disposable directory**

```bash
PEG_LOTTIE_WORK=$(mktemp -d -t peg-bench-lottie.XXXXXX)
npx degit diffusionstudio/lottie "$PEG_LOTTIE_WORK"
cd "$PEG_LOTTIE_WORK"
npm install
mkdir -p public/projects/peg-bench/scene-1
```

Use `public/projects/peg-bench/scene-1/lottie.json` as the authoritative scene. Include top-level `v`, `fr`, `ip`, `op`, `w`, `h`, `nm`, `assets`, and `layers`. Build six horizontal evidence rows with neutral origin nodes, two grey unchanged connectors, four teal changed connectors, and brick final emphasis. Use trim-path reveals followed by endpoint opacity/scale settles. Use calm ease-out timing and no bounce. Keep all scientific labels in adjacent HTML rather than inside the animation.

- [ ] **Step 4: Validate and visually inspect the scene in Skottie**

Run `npm run dev`, capture the printed localhost port as `PEG_LOTTIE_PORT`, then inspect:

```bash
node -e "JSON.parse(require('fs').readFileSync('public/projects/peg-bench/scene-1/lottie.json','utf8'))"
curl -fsS "http://127.0.0.1:${PEG_LOTTIE_PORT}/__context"
```

Open and capture `/peg-bench/scene-1?frame=0`, `?frame=36`, and `?frame=71`. Verify frame 0 is intentionally quiet but non-corrupt, frame 36 communicates path changes in progress, frame 71 is a balanced and complete figure, and no element clips or overlaps.

- [ ] **Step 5: Export the scene and vendor a pinned light runtime**

From the public clone, while retaining the `PEG_LOTTIE_WORK` value from the setup shell:

```bash
cp "$PEG_LOTTIE_WORK/public/projects/peg-bench/scene-1/lottie.json" docs/assets/peg-ranking-trace.json
mkdir -p /tmp/peg-lottie-runtime docs/assets/vendor
npm pack lottie-web@5.13.0 --pack-destination /tmp/peg-lottie-runtime
tar -xzf /tmp/peg-lottie-runtime/lottie-web-5.13.0.tgz -C /tmp/peg-lottie-runtime
cp /tmp/peg-lottie-runtime/package/build/player/lottie_light.min.js docs/assets/vendor/lottie-light.min.js
cp /tmp/peg-lottie-runtime/package/LICENSE.md docs/assets/vendor/lottie-web-LICENSE.md
```

- [ ] **Step 6: Add the static-first figure and enhancement controller**

The evidence figure must use this state model:

```html
<figure class="ranking-figure" aria-labelledby="ranking-caption">
  <div id="ranking-motion" class="motion-stage" data-lottie-src="./assets/peg-ranking-trace.json" data-motion-state="fallback">
    <img class="motion-fallback" src="./assets/peg-ranking-shift.svg" width="2500" height="2475" alt="Six-panel comparison showing four active leader identity changes after 3 bps accounting">
    <div class="motion-canvas" aria-hidden="true"></div>
  </div>
  <figcaption id="ranking-caption"><strong>Four of six active selections change identity.</strong> LightGBM leads all R0 panels; Kairos-64 leads four clean-active R4 panels at 3 bps.</figcaption>
</figure>
```

Load `lottie-light.min.js` with `defer` before `site.js`. In `site.js`, preserve the year update, close the mobile `details` after anchor activation, skip enhancement when reduced motion is requested, load the scene with `renderer: "svg"`, `loop: false`, and `autoplay: false`, set `data-motion-state="ready"` on `DOMLoaded`, play once through an `IntersectionObserver` threshold of 0.45, and restore `fallback` on `data_failed` or `error`. CSS hides the fallback only in the `ready` state and restores it in reduced-motion mode.

- [ ] **Step 7: Run animation tests and inspect local playback**

Run:

```bash
node --test site_tests/site-contract.test.mjs
node -e "JSON.parse(require('fs').readFileSync('docs/assets/peg-ranking-trace.json','utf8'))"
python3 -m http.server 4173 --directory docs >/tmp/peg-site-server.log 2>&1 &
PEG_SITE_PID=$!
npx playwright screenshot --viewport-size=390,844 http://127.0.0.1:4173/#results /tmp/peg-motion-mobile.png
npx playwright screenshot --viewport-size=1280,800 http://127.0.0.1:4173/#results /tmp/peg-motion-desktop.png
kill "$PEG_SITE_PID"
git diff --check
```

Expected: tests and JSON parsing pass; screenshots show a nonblank final or in-progress figure; the static SVG remains visible when JavaScript is disabled or reduced motion is emulated.

- [ ] **Step 8: Commit the motion enhancement**

```bash
git add docs/index.html docs/site.js docs/styles.css docs/assets/peg-ranking-trace.json docs/assets/vendor site_tests/site-contract.test.mjs
git commit -m "feat: add evidence trace motion with static fallback"
```

---

### Task 6: Finish Metadata, Social Preview, And The Error Page

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/404.html`
- Create: `site_tools/social-card.html`
- Create: `docs/assets/peg-bench-social.png`
- Modify: `site_tests/site-contract.test.mjs`

**Interfaces:**
- Consumes: poster title, thesis, palette, ranking SVG, canonical Pages URL.
- Produces: anonymous ScholarlyArticle metadata, a 1200x630 poster-matched preview, and a consistent 404 recovery path.

- [ ] **Step 1: Add failing metadata, social image, and 404 tests**

Append:

```js
test("metadata and error recovery match the anonymous poster identity", () => {
  const html = read("docs/index.html");
  const notFound = read("docs/404.html");
  assert.match(html, /<title>When Prediction Leaderboards Can Fail \| PEG-Bench<\/title>/);
  assert.match(html, /property="og:image" content="https:\/\/computational-decision-lab\.github\.io\/peg-bench\/assets\/peg-bench-social\.png"/);
  assert.match(html, /"@type":\s*"ScholarlyArticle"/);
  assert.doesNotMatch(html, /"author"|"affiliation"|"publisher"/i);
  assert.match(notFound, /Return to PEG-Bench/);
  assert.ok(exists("docs/assets/peg-bench-social.png"));
  const png = readFileSync(resolve(root, "docs/assets/peg-bench-social.png"));
  assert.equal(png.readUInt32BE(16), 1200);
  assert.equal(png.readUInt32BE(20), 630);
});
```

- [ ] **Step 2: Run the metadata test and verify it fails**

Run: `node --test --test-name-pattern="metadata and error" site_tests/site-contract.test.mjs`

Expected: FAIL on the title, new social preview path, and missing preview PNG.

- [ ] **Step 3: Update metadata and create the social-card source**

Update title, description, theme color, canonical URL, Open Graph/Twitter tags, and anonymous ScholarlyArticle JSON-LD. Do not add Person or Organization objects.

Create `site_tools/social-card.html` as a fixed 1200x630 poster composition using the exact palette. It contains only `PEG-Bench`, `When Prediction Leaderboards Can Fail`, `Prediction rankings can reverse after costs, turnover, and risk.`, a simplified six-row leader trace, and `4 / 6 leader identities change at 3 bps`.

- [ ] **Step 4: Capture the exact social image and update 404 recovery**

```bash
npx playwright screenshot --viewport-size=1200,630 "file://$(pwd)/site_tools/social-card.html" docs/assets/peg-bench-social.png
```

Update `docs/404.html` to reuse `styles.css`, the wordmark, poster title, a short not-found message, `Return to PEG-Bench`, `Read paper`, and `Open GitHub` links. Keep it anonymous and functional without JavaScript.

- [ ] **Step 5: Run metadata tests and inspect the preview**

Run:

```bash
node --test site_tests/site-contract.test.mjs
file docs/assets/peg-bench-social.png
git diff --check
```

Expected: all tests pass; `file` reports a 1200x630 PNG; visual inspection shows no clipping or hidden identity text.

- [ ] **Step 6: Commit metadata and recovery work**

```bash
git add docs/index.html docs/404.html docs/assets/peg-bench-social.png site_tools/social-card.html site_tests/site-contract.test.mjs
git commit -m "chore: complete anonymous metadata and social preview"
```

---

### Task 7: Audit Accessibility, Motion, Performance, And Every Viewport

**Files:**
- Modify when findings require: `docs/index.html`
- Modify when findings require: `docs/styles.css`
- Modify when findings require: `docs/site.js`
- Modify when findings require: `site_tests/site-contract.test.mjs`

**Interfaces:**
- Consumes: complete local site from Tasks 1-6.
- Produces: verified responsive behavior, accessible interaction, complete static fallback, clean design-skill audit, and final local screenshots.

- [ ] **Step 1: Run deterministic tests and repository scans**

Run:

```bash
node --test site_tests/site-contract.test.mjs
git diff --check
rg -in "Ke Wang|Yangyang Liang|Xiaorui Tang|Georgia Tech|Guangdong University|Hong Kong University|HKUST|KDD|MLF 2026|NeurIPS" docs --glob '!assets/*.pdf'
rg -n "linear-gradient|radial-gradient|backdrop-filter|filter:\s*blur" docs/styles.css docs/site.js
```

Expected: tests pass, whitespace is clean, and both searches return no matches.

- [ ] **Step 2: Run the Impeccable detector on every changed frontend target**

Run:

```bash
node /opt/projects/research/peg/.agents/skills/impeccable/scripts/detect.mjs --json docs/index.html docs/styles.css docs/site.js docs/404.html
```

Expected: no blocking accessibility, hierarchy, responsive, motion, or anti-pattern findings. Fix any concrete finding in the smallest owning file and rerun the detector.

- [ ] **Step 3: Start the site and capture all required viewports**

```bash
python3 -m http.server 4173 --directory docs >/tmp/peg-site-server.log 2>&1 &
PEG_SITE_PID=$!
for size in 320,568 390,844 430,932 768,1024 1280,800 1440,900; do
  name=${size/,/x}
  npx playwright screenshot --full-page --viewport-size="$size" http://127.0.0.1:4173/ "/tmp/peg-site-${name}.png"
done
kill "$PEG_SITE_PID"
```

Inspect each screenshot for overlap, clipping, blank media, hidden actions, tiny table text, unbounded line length, accidental cards, broken hierarchy, and horizontal page overflow. At 390x844, confirm the first viewport contains the identity, thesis, decisive result, and at least one paper or evidence action.

- [ ] **Step 4: Exercise keyboard, no-JavaScript, failed-animation, and reduced-motion paths**

Use Chromium to verify:

- skip link becomes visible on focus and reaches `#main`;
- mobile disclosure opens with Enter/Space, exposes all anchors, and closes after selection;
- every focus ring is visible on paper and dark surfaces;
- disabling JavaScript leaves navigation, evidence, tables, application disclosure, and materials usable;
- blocking `peg-ranking-trace.json` leaves `peg-ranking-shift.svg` visible;
- reduced motion prevents entrance and Lottie playback and shows the final static state;
- 200 percent zoom does not hide content or controls.

- [ ] **Step 5: Verify local links, local asset MIME types, and payload sizes**

Run:

```bash
python3 -m http.server 4173 --directory docs >/tmp/peg-site-server.log 2>&1 &
PEG_SITE_PID=$!
curl -fsSI http://127.0.0.1:4173/
curl -fsSI http://127.0.0.1:4173/assets/peg-bench-technical-report.pdf
curl -fsSI http://127.0.0.1:4173/assets/peg-ranking-shift.svg
curl -fsSI http://127.0.0.1:4173/assets/peg-ranking-trace.json
curl -fsSI http://127.0.0.1:4173/assets/peg-bench-social.png
du -h docs/assets/vendor/lottie-light.min.js docs/assets/peg-ranking-trace.json docs/assets/fonts/Lato-*.ttf
kill "$PEG_SITE_PID"
```

Expected: all responses are 200 with appropriate content types; the runtime and animation remain small enough to defer without blocking the static content; fonts use `font-display: swap`.

- [ ] **Step 6: Run the animation review and final regression suite**

Review the scene against the text-to-lottie checks: reading-order reveal, calm research easing, no bounce, readable midpoint, clean final settle, no cropped strokes, and no decorative motion. Then run:

```bash
node --test site_tests/site-contract.test.mjs
python -m unittest discover -s tests -v
git status --short
```

Expected: site tests and existing repository tests pass. Only intentional frontend changes are present.

- [ ] **Step 7: Commit any audit fixes**

```bash
git add docs/index.html docs/styles.css docs/site.js docs/404.html site_tests/site-contract.test.mjs
git commit -m "fix: polish responsive and accessible research reading"
```

Skip this commit only if the audit produces no source changes.

---

### Task 8: Push And Verify GitHub Pages End To End

**Files:**
- Create: `docs/assets/site-checksums.sha256`
- Verify: `.github/workflows/pages.yml`

**Interfaces:**
- Consumes: locally verified `main` branch.
- Produces: deployed GitHub Pages site at `https://computational-decision-lab.github.io/peg-bench/` with verified live asset parity and links.

- [ ] **Step 1: Refresh public asset checksums without touching research-result files**

Create a site-specific checksum file without changing the research artifact's root `SHA256SUMS`, then verify:

```bash
sha256sum docs/index.html docs/styles.css docs/site.js docs/404.html docs/assets/peg-evaluation-chain.svg docs/assets/peg-ranking-shift.svg docs/assets/peg-ranking-trace.json docs/assets/peg-bench-social.png > docs/assets/site-checksums.sha256
sha256sum -c docs/assets/site-checksums.sha256
git diff --check
```

Expected: every listed file reports `OK`.

- [ ] **Step 2: Run the complete pre-push gate**

```bash
node --test site_tests/site-contract.test.mjs
python -m unittest discover -s tests -v
git status --short
git log --oneline -8
```

Expected: all tests pass; status contains only the intentional site checksum file if not yet committed; recent commits correspond to Tasks 1-7.

- [ ] **Step 3: Commit checksums and push the isolated public clone**

```bash
git add docs/assets/site-checksums.sha256
git commit -m "chore: refresh public site checksums"
git push origin main
```

- [ ] **Step 4: Watch the Pages workflow to completion**

```bash
PEG_PAGES_RUN_ID=$(gh run list --repo computational-decision-lab/peg-bench --workflow pages.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$PEG_PAGES_RUN_ID" --repo computational-decision-lab/peg-bench --exit-status
```

Expected: the newest `Deploy research site` workflow completes successfully.

- [ ] **Step 5: Verify live endpoints, links, MIME types, and content**

```bash
curl -fsSI https://computational-decision-lab.github.io/peg-bench/
curl -fsSI https://computational-decision-lab.github.io/peg-bench/assets/peg-bench-technical-report.pdf
curl -fsSI https://computational-decision-lab.github.io/peg-bench/assets/peg-ranking-shift.svg
curl -fsSI https://computational-decision-lab.github.io/peg-bench/assets/peg-ranking-trace.json
curl -fsSI https://computational-decision-lab.github.io/peg-bench/assets/peg-bench-social.png
curl -fsS https://computational-decision-lab.github.io/peg-bench/ | rg "When Prediction Leaderboards Can Fail|Illustrative transfer scenarios|computational-decision-lab/peg-bench"
```

Expected: HTML, PDF, SVG, JSON, and PNG endpoints return 200 with correct content types; the live page contains the poster title, transfer disclosure, and renamed repository links.

- [ ] **Step 6: Verify live/local asset parity and capture final live screenshots**

Download the live HTML, CSS, JavaScript, ranking SVG, Lottie JSON, and social PNG to a temporary directory, compare SHA-256 values with local files, and capture 390x844 and 1440x900 screenshots from the live URL.

Expected: local and live hashes match; screenshots are nonblank, poster-aligned, mobile-readable, and free of overlap or hidden content.

- [ ] **Step 7: Record the deployment evidence**

Report the final commit, Pages workflow run ID, live URL, tested viewport list, site-test result, repository-test result, and live asset parity. Do not claim deployment complete without the workflow and live endpoint evidence.
