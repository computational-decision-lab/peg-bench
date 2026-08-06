# PEG-Bench Controlled Contract Figure Integration

**Status:** Approved direction A on 2026-08-06

## Objective

Integrate the current controlled diagnostic contract figure into the PEG-Bench
research site without repeating the existing thesis, mechanism, or evidence
graphics. Preserve the site's reviewer-oriented reading order and its
comfortable mobile presentation.

## Figure Decisions

### Controlled diagnostic contract

Use `figure1_peg_bench_contract.png` as the only newly published figure. It is
consistent with the released six-panel evidence: LightGBM leads all populated
R0 panels, four of six clean-active identities change at 3 bps, positive point
estimates remain uncertified by their intervals, and the supported conclusion
is selection-conclusion drift rather than trading alpha.

Place the figure at the end of `#design`, where it connects the frozen contract
to the later report card. It replaces the visible compact R0-R4 ladder; it does
not appear again in the hero, results, or materials grid as a second thumbnail.
The existing contract table remains the canonical text and mobile reading
surface.

### Appendix interface figure

Do not publish `peg_bench_appendix.png` on the main site. Its shared wrapper,
report card, and optional OG-PA branch overlap the mechanism and limitations
sections, while OG-PA is not the main benchmark claim. The technical report
already provides the appropriate appendix context.

### Thesis overview figure

Do not publish `peg_bench_thesis_overview.png`. Its five-stage thesis overlaps
the current hero chain and ranking animation, and its illustrative models and
single numerical example are not the site's current six-panel evidence.
Remove the unreferenced legacy copy from `docs/assets/`.

## Responsive Presentation

- At widths of 768 CSS pixels and above, show the controlled contract figure as
  a full-width, ruled research figure after the frozen contract table.
- Below 768 CSS pixels, hide the inline raster preview and keep the existing
  stacked semantic contract table as the primary reading view.
- Provide one 44-pixel-tall text link, adjacent to the contract table, that
  opens the full-resolution PNG in a new tab on every viewport.
- Do not introduce horizontal page scrolling, an image carousel, a modal, or a
  second mobile-specific raster.
- Keep the image's intrinsic `1916 x 821` dimensions and lazy-load it because it
  appears below the fold.

## Caption And Accessibility

The visible caption identifies the image as the controlled diagnostic contract
and states that it connects frozen data, a shared action interface, the R0-R4
diagnostic, the report card, and the bounded conclusion.

The image alt text conveys the same sequence and its bounded result without
claiming profitability. The full-resolution link has a clear accessible name.
The semantic table remains available at every viewport, including to assistive
technology.

## Public And Scientific Boundaries

- Do not add author names, affiliations, school names, venue names, workshop
  names, or venue-year labels to visible copy, metadata, alt text, or filenames.
- Do not add raw vendor L2 data, reconstructable features, processed tensors,
  credentials, or private build artifacts.
- Do not change any published result, confidence interval, paper PDF, or public
  artifact table.
- Publish the selected figure under the neutral filename
  `docs/assets/peg-controlled-diagnostic-contract.png`.

## Verification

- Extend the static site contract tests before implementation and observe the
  new test fail because the selected asset and markup are absent and the legacy
  figure remains.
- Verify the new asset hash matches the approved local source exactly.
- Verify no HTML reference or tracked site asset includes either rejected
  figure.
- Run the complete Node site contract suite and repository whitespace checks.
- Inspect 320x568, 390x844, 430x932, 768x1024, 1280x800, and 1440x900 with
  Playwright; assert no horizontal overflow and confirm the image/link visibility
  rules.
- Run keyboard, no-JavaScript, reduced-motion, broken-link, privacy-term, and
  local/live asset parity checks before declaring deployment complete.
