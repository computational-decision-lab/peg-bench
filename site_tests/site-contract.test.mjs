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
  assert.match(design, /960x360/);
  assert.match(design, /ip:\s*0/);
  assert.match(design, /op:\s*72/);
  assert.match(design, /frames 0-71/);
  assert.match(design, /static SVG fallback/i);
});

export { exists, read, root };
