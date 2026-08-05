import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const read = (path) => readFileSync(resolve(root, path), "utf8");
const exists = (path) => existsSync(resolve(root, path));
const attributes = (html, name) => [...html.matchAll(new RegExp(`${name}=["']([^"']+)["']`, "g"))].map((match) => match[1]);

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
  assert.match(design, /960x360/);
  assert.match(design, /ip:\s*0/);
  assert.match(design, /op:\s*72/);
  assert.match(design, /frames 0-71/);
  assert.match(design, /static SVG fallback/i);
});

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

test("404 page uses the current responsive navigation and button classes", () => {
  const html = read("docs/404.html");
  assert.match(html, /class="nav-primary"/);
  assert.match(html, /class="nav-paper"/);
  assert.match(html, /class="button button-primary"/);
  assert.doesNotMatch(html, /\b(?:nav-links|nav-button|button-dark)\b/);
});

test("small text on soft bands uses an AA poster color", () => {
  const css = read("docs/styles.css");
  const selectors = [
    ".reader-map a span",
    ".numbered-heading-light > span",
    ".concept-index",
    ".transfer-disclosure",
    ".boundary-band .text-link"
  ];
  for (const selector of selectors) {
    const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const rule = css.match(new RegExp(`${escaped}\\s*\\{[^}]*\\}`))?.[0];
    assert.ok(rule, `missing CSS rule for ${selector}`);
    assert.match(rule, /color:\s*var\(--(?:ink|teal)\)/, `${selector} is not AA on --soft`);
  }
});

test("table wrappers do not advertise obsolete scrolling", () => {
  const html = read("docs/index.html");
  for (const className of ["design-table-wrap", "mechanism-table-wrap", "result-table-wrap", "limits-table-wrap"]) {
    const wrapper = html.match(new RegExp(`<div class="${className}"[^>]*>`))?.[0];
    assert.ok(wrapper, `missing ${className}`);
    assert.doesNotMatch(wrapper, /\btabindex=/, `${className} is unnecessarily focusable`);
    assert.doesNotMatch(wrapper, /\baria-label=/, `${className} advertises obsolete scrolling`);
  }
});

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

export { exists, read, root };
