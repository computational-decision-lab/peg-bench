# Design System

## Direction

The site is the mobile-first research companion to the PEG-Bench poster: editorial-scientific, precise, restrained, and evidence-led. It is an asymmetric Swiss-style composition with strong left alignment and deliberate whitespace. Thin rules and alignment replace decorative cards and shadows; flat evidence graphics replace stock imagery, abstract gradients, and ornamental illustrations.

It is not a product landing page, trading dashboard, or conference microsite. Do not use gradients, glows, glass effects, or hue changes between sections.

## Palette

- `--paper`: #F8F7F2, the poster-matched substrate.
- `--ink`: #152536, for primary text and dark bands.
- `--teal`: #0B6E69, for supported execution paths and evidence navigation.
- `--brick`: #B8493C, for reversals, negative outcomes, cautions, and boundary labels.
- `--grey`: #5E6B76, for secondary text.
- `--rule`: #CDD9DF, for the single divider treatment.
- `--soft`: #ECEEEA, for restrained evidence backgrounds.
- `--white`: #FFFFFF, for accessible contrast where the paper tone is insufficient.

Teal and brick are semantic colors, not competing decoration.

## Typography

Self-host Lato in the weights needed for body, headings, and numerical emphasis, matching the poster and avoiding a third-party font request. Use a system monospace stack only for protocol labels, compact values, and file paths.

- Desktop body: 17px with 1.65 line height.
- Mobile body: 17px with 1.65 line height.
- Prose measure: 58ch to 68ch.
- Use discrete breakpoint sizes, never viewport-width font scaling.
- Headlines are sentence case with zero letter spacing. Uppercase is reserved for short protocol labels.

## Geometry And Rules

Square geometry is the default. Controls may use a 4px radius for focus and tactile clarity; content panels remain unrounded. Use one hairline rule system through `--rule`; do not substitute rounded cards, heavy borders, or shadows for hierarchy.

## Information Architecture

Keep this eight-section reading order and anchor intent:

1. `#problem`: Real-world problem.
2. `#concept`: Plain-language concept.
3. `#design`: Experiment design.
4. `#mechanism`: Mechanism interaction.
5. `#results`: Evidence results.
6. `#limits`: Limitations.
7. `#engineering`: Engineering application and transfer scenarios.
8. `#materials`: Paper, code, evidence, and reproduction materials.

Show this map once near the beginning as an unframed horizontal index on desktop and a compact two-column list on mobile. Section numbers are wayfinding, not oversized decorative badges.

## Layout And Responsive Behavior

Use a sticky desktop navigation with the PEG-Bench wordmark, `Problem`, `Design`, `Evidence`, `Limits`, `Materials`, and one primary `Paper` action. Mobile uses a compact sticky bar with the wordmark, `Paper`, and a native disclosure menu. All targets are at least 44 by 44 CSS pixels, and anchored sections account for the sticky-header offset.

The hero is the scan landing state: it contains the project marker, the thesis, one decisive result, a paper or evidence action, and a visible hint of the next section. Preserve stable aspect ratios for evidence graphics. Use mobile-specific evidence compositions rather than scaled-down desktop canvases, and do not allow horizontal page overflow.

Validate at 320x568, 390x844, 430x932, 768x1024, 1280x800, and 1440x900. Mobile side padding is 18px at 320px and 22px at 390px and above. Hero actions keep one line per label and stack when necessary. Long words, paths, confidence intervals, and numerical labels wrap or use controlled overflow without clipping.

Below 720px, tables transform into readable stacked records rather than miniature desktop tables. The experiment matrix becomes records labeled `Dimension`, `Frozen specification`, and `Why it matters`. The six-panel evidence table becomes records ordered by panel, R0 leader and RankIC, R4 active leader and Sharpe, interval, and result label; its semantic table remains available to assistive technology. Safe-area insets protect sticky header and footer actions, and below-fold images lazy-load with explicit intrinsic dimensions.

## Motion

Motion explains evidence and hierarchy; it never decorates empty space. The only evidence animation is a transparent Lottie scene for the poster-derived R0-to-R4 leader trace: a 960x360 canvas, 30 fps, 2.4 seconds, exact frame bounds `ip: 0`, `op: 72` (frames 0-71), and no loop. It reveals six neutral R0 origins, draws staged ease-out connectors, resolves unchanged paths in grey and changed paths in teal, then reveals `4 / 6 changed` in brick. Do not animate numerical values, confidence intervals, or scientific labels into misleading intermediate states.

The animation runs once when substantially visible. A static SVG fallback with identical final-state information is present in the markup before the player loads and remains available when JavaScript, the runtime, or motion is unavailable. Under `prefers-reduced-motion: reduce`, disable orchestration and Lottie playback and display the static final state.

Hero orchestration uses one 360-520ms opacity/translate reveal per load. Navigation and buttons use 120-180ms compositor-only feedback. The mobile disclosure uses native state change with a short opacity/transform transition. Do not use infinite animation, scroll-jacking, parallax, blur animation, or layout-property animation.

## Accessibility And Resilience

Maintain semantic landmarks, a skip link, one H1, logical heading order, figure captions, table captions, row and column headers, and descriptive alt text. Meet WCAG 2.1 AA contrast, including focus indicators and small labels. Use `:focus-visible` with an offset visible on paper and dark surfaces; never use color alone to distinguish states. Keep core content and links usable without JavaScript, including the static SVG fallback, and preserve readable zoom behavior through 200 percent.
