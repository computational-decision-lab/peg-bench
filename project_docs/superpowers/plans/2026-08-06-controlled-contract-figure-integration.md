# Controlled Contract Figure Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the approved controlled diagnostic contract figure in the experiment-design section while removing the obsolete thesis overview asset and preserving a readable mobile contract view.

**Architecture:** Keep the semantic contract table as the canonical cross-device content. Add one desktop/tablet research figure and one full-resolution text link, hide only the raster preview below 768px, and remove the existing visible compact ladder plus the unreferenced legacy PNG.

**Tech Stack:** Semantic HTML, CSS media queries, Node.js built-in test runner, Playwright, GitHub Pages.

## Global Constraints

- Work only in `/tmp/peg-bench-publish.cCNpCi`, the isolated public clone.
- Publish only `figure1_peg_bench_contract.png` under the neutral filename `docs/assets/peg-controlled-diagnostic-contract.png`.
- Do not publish the appendix interface figure or thesis overview figure.
- Do not change released values, paper PDFs, artifact tables, or public-data boundaries.
- Do not add author, affiliation, school, venue, workshop, or venue-year text.
- Keep the semantic contract table available on every viewport and to assistive technology.
- Hide the inline raster preview below 768px; provide the full-resolution link at every width.
- Keep all controls at least 44 CSS pixels high and prevent horizontal page overflow.
- Validate 320x568, 390x844, 430x932, 768x1024, 1280x800, and 1440x900.

---

### Task 1: Encode The Figure Selection And Responsive Contract

**Files:**
- Modify: `site_tests/site-contract.test.mjs`
- Modify: `DESIGN.md`

**Interfaces:**
- Consumes: `project_docs/superpowers/specs/2026-08-06-controlled-contract-figure-design.md`.
- Produces: failing assertions for the selected asset, rejected assets, semantic markup, and responsive CSS behavior.

- [ ] **Step 1: Add a failing site contract test**

Append this test before the final export:

```js
test("controlled contract figure is unique, public-safe, and mobile-adapted", () => {
  const html = read("docs/index.html");
  const css = read("docs/styles.css");
  const design = read("DESIGN.md");
  const selected = "docs/assets/peg-controlled-diagnostic-contract.png";
  const legacy = "docs/assets/peg-bench-thesis-overview.png";

  assert.ok(exists(selected), `missing ${selected}`);
  assert.ok(!exists(legacy), `obsolete duplicate remains: ${legacy}`);
  assert.equal((html.match(/peg-controlled-diagnostic-contract\.png/g) || []).length, 2,
    "the selected figure must appear once as an image and once as its full-resolution link");
  assert.doesNotMatch(html, /peg-bench-thesis-overview\.png|peg-bench-appendix\.png/);
  assert.match(html, /class="contract-figure"/);
  assert.match(html, /width="1916" height="821" loading="lazy"/);
  assert.match(html, /Open the full-resolution contract figure/);
  assert.doesNotMatch(html, /class="ladder-inline"/);
  assert.match(css, /@media \(max-width: 767px\)[\s\S]*\.contract-figure[\s\S]*display:\s*none/);
  assert.match(design, /controlled diagnostic contract/i);
});
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
node --test --test-name-pattern="controlled contract figure" site_tests/site-contract.test.mjs
```

Expected: FAIL with `missing docs/assets/peg-controlled-diagnostic-contract.png`.

- [ ] **Step 3: Document the approved responsive figure rule**

Add this paragraph to the experiment-design portion of `DESIGN.md`:

```markdown
The controlled diagnostic contract is the only full-width overview figure in
the experiment-design section. It replaces the compact visible R0-R4 ladder at
768px and above. Below 768px, the stacked semantic contract table remains the
primary reading view and a 44px text link opens the full-resolution figure; the
wide raster preview is not scaled into unreadable mobile text.
```

- [ ] **Step 4: Keep the focused test red for the missing implementation**

Run the same focused test.

Expected: FAIL on the missing selected asset; the design-document assertion now passes.

---

### Task 2: Curate The Single Public Figure And Remove The Legacy Duplicate

**Files:**
- Create: `docs/assets/peg-controlled-diagnostic-contract.png`
- Delete: `docs/assets/peg-bench-thesis-overview.png`

**Interfaces:**
- Consumes: `/opt/projects/research/peg/kdd-mlf-2026/figures/figure1_peg_bench_contract.png` with approved SHA-256 `d23bb8655583e1044e577ed5b65aaec6c2b61a5debe4200799b3178456fbe0ae`.
- Produces: one byte-identical public PNG with intrinsic dimensions 1916x821.

- [ ] **Step 1: Copy only the approved binary and remove the tracked legacy binary**

Run:

```bash
cp /opt/projects/research/peg/kdd-mlf-2026/figures/figure1_peg_bench_contract.png docs/assets/peg-controlled-diagnostic-contract.png
git rm docs/assets/peg-bench-thesis-overview.png
```

- [ ] **Step 2: Verify byte identity, dimensions, and rejection boundaries**

Run:

```bash
sha256sum /opt/projects/research/peg/kdd-mlf-2026/figures/figure1_peg_bench_contract.png docs/assets/peg-controlled-diagnostic-contract.png
file docs/assets/peg-controlled-diagnostic-contract.png
find docs/assets -maxdepth 1 -type f \( -iname '*appendix*' -o -iname '*thesis-overview*' \)
```

Expected: both hashes are identical; `file` reports `1916 x 821`; the rejected-asset search prints nothing.

- [ ] **Step 3: Run the focused test and confirm it now fails on absent markup**

Run the focused Node test.

Expected: FAIL because `docs/index.html` does not yet contain `contract-figure`.

---

### Task 3: Integrate The Figure Without Repeating The Reading Path

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/styles.css`

**Interfaces:**
- Consumes: the curated PNG from Task 2 and existing `.design-table-wrap` markup.
- Produces: one inline desktop/tablet figure, one all-viewport full-resolution link, and no visible compact ladder.

- [ ] **Step 1: Replace the compact ladder in `#design`**

Replace the complete existing `<div class="ladder-inline">` block with:

```html
<figure class="contract-figure">
  <img src="./assets/peg-controlled-diagnostic-contract.png" width="1916" height="821" loading="lazy" alt="Controlled diagnostic contract connecting a frozen 90-day six-panel data design, shared action interface, R0-to-R4 diagnostic, unified report card, and the bounded result that four of six clean-active leader identities change at 3 bps without establishing trading alpha">
  <figcaption><strong>Controlled diagnostic contract</strong><span>Frozen data and interfaces lead to one traceable report card and a bounded selection conclusion.</span></figcaption>
</figure>
```

Add this link to the existing design-section `.inline-links` list:

```html
<a href="./assets/peg-controlled-diagnostic-contract.png" target="_blank" rel="noreferrer">Open the full-resolution contract figure</a>
```

- [ ] **Step 2: Add restrained figure layout and mobile art-direction rules**

Add:

```css
.contract-figure {
  margin: 48px 0 0 calc((100% + 24px) / 12);
  padding-top: 16px;
  border-top: 1px solid var(--rule);
}

.contract-figure img {
  width: 100%;
  height: auto;
  aspect-ratio: 1916 / 821;
  object-fit: contain;
  background: var(--white);
}

.contract-figure figcaption {
  align-items: start;
}

@media (max-width: 767px) {
  .contract-figure {
    display: none;
  }
}
```

- [ ] **Step 3: Run focused and full contract tests**

Run:

```bash
node --test --test-name-pattern="controlled contract figure" site_tests/site-contract.test.mjs
node --test site_tests/site-contract.test.mjs
git diff --check
```

Expected: focused test passes; all site tests pass; whitespace check is clean.

- [ ] **Step 4: Commit the tested integration**

```bash
git add DESIGN.md site_tests/site-contract.test.mjs docs/index.html docs/styles.css docs/assets/peg-controlled-diagnostic-contract.png docs/assets/peg-bench-thesis-overview.png
git commit -m "feat: add controlled diagnostic contract figure"
```

---

### Task 4: Verify Presentation, Integrity, And Deployment

**Files:**
- Modify: `docs/assets/site-checksums.sha256`

**Interfaces:**
- Consumes: final `docs/` tree from Task 3.
- Produces: responsive screenshots, integrity manifest, deployed asset parity, and live HTTP evidence.

- [ ] **Step 1: Start a local static server and run responsive Playwright checks**

Use an available local port and verify all required viewports. For each viewport assert:

```js
document.documentElement.scrollWidth === document.documentElement.clientWidth
```

Also assert `.contract-figure` is `display: none` at 320, 390, and 430 widths; it is visible at 768, 1280, and 1440 widths; the full-resolution link is visible and at least 44px tall at every width.

- [ ] **Step 2: Capture and inspect representative screenshots**

Capture 390x844 around `#design` and 1440x900 around `#design`. Confirm the mobile view uses stacked records with no tiny raster text, the desktop image is not clipped, captions do not overlap, and the next section remains visually distinct.

- [ ] **Step 3: Run the Impeccable mechanical detector once**

Run:

```bash
node /opt/projects/research/peg/.agents/skills/impeccable/scripts/detect.mjs --json docs/index.html
```

Resolve any finding introduced by this change, then rerun the full Node test suite and Playwright checks if production files changed.

- [ ] **Step 4: Refresh and verify the site integrity manifest**

Run:

```bash
sha256sum docs/index.html docs/styles.css docs/site.js docs/404.html docs/assets/peg-bench-paper.pdf docs/assets/peg-bench-technical-report.pdf docs/assets/peg-evaluation-chain.svg docs/assets/peg-ranking-shift.svg docs/assets/peg-ranking-trace.json docs/assets/peg-bench-social.png docs/assets/peg-controlled-diagnostic-contract.png > docs/assets/site-checksums.sha256
sha256sum -c docs/assets/site-checksums.sha256
```

Expected: every entry reports `OK`.

- [ ] **Step 5: Commit verification metadata, merge, and push**

Commit the refreshed checksum manifest, merge the feature branch into `main`, push `main` to `origin`, and wait for the Pages workflow to complete successfully.

- [ ] **Step 6: Verify the live deployment**

Fetch the canonical page and selected PNG. Verify HTTP 200, `image/png`, the live HTML contains exactly one image plus one link reference to the new asset, rejected asset names are absent, all public links resolve, and the live PNG SHA-256 matches the local source.
