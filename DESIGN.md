# Design System

## Direction

Research instrument, not startup landing page. The visual voice is a warm paper substrate with ink-black typography and deliberate coral/teal evidence marks. Dense enough for a reviewer, paced enough for a first-time visitor.

## Palette

- `--ink`: #101722
- `--ink-soft`: #334155
- `--paper`: #f5f1e9
- `--paper-deep`: #e8e2d7
- `--white`: #fffdf8
- `--coral`: #e85b52
- `--teal`: #2f8f83
- `--yellow`: #e2b94f
- `--line`: rgba(16, 23, 34, 0.16)

No gradients. Coral marks prediction-to-execution failure; teal marks an evidence path; yellow is reserved for caution and release boundaries.

## Typography

Use a system sans stack for readable long-form text and a compact system monospace stack only for values, paths, and protocol labels. Headings use strong weight and tight measure rather than decorative type.

## Layout

Full-width color bands with a centered reading column. Hero uses a two-column evidence composition on desktop and stacks on mobile. Repeated evidence is presented as restrained bordered rows, not nested cards. Maximum content width is 1180px; text measure is 66ch.

## Components

- Sticky navigation with text links and one primary paper action.
- Hero thesis statement with two evidence numbers and the overview figure.
- R0-to-R4 ladder as a horizontal sequence that wraps on narrow screens.
- Six-panel evidence table with explicit winner shifts and post-cost Sharpe.
- Release boundary callout with direct links to artifact and reproduction script.
- Footer with paper, GitHub, license, and scope links.

## Motion

No required motion. A short opacity/translate entrance is allowed for the hero only and is disabled under `prefers-reduced-motion`. Interactive controls use compositor-only transitions under 180ms.
