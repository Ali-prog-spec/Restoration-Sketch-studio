# Google Stitch — design guide (MANUAL ACTION REQUIRED)

The assignment (PDF p.2, p.8) requires the interface design — page structure, colours,
controls, cards, image panels, responsive layout — to be **developed first in Google Stitch**
(https://stitch.withgoogle.com), with evidence in the report.

**Status (2026-10-03): PARTLY DONE.** The student generated the designs in Stitch and added them to
`report/stitch/`:

| File | Content |
|---|---|
| `stitch_logo.png` | app logo (also used as `frontend/public/logo.png`) |
| `stitch_universal.png` | Task 1 Universal Restoration screen |
| `stitch_hard_routing.png` | Task 2 Hard-Routed Restoration screen |
| `stitch_soft_moe.png` | Task 3 Soft MoE screen |
| `stitch_face_to_sketch.png` | Task 4 Face-to-Sketch screen |
| `stitch_system_status.png` | System status screen |

Still missing (recommended): a screenshot of the Stitch project / prompt history (`stitch_prompt.png`)
as proof of origin.

The React frontend was restyled to follow these designs (layout, sidebar, header, cards, monospace
labels, tagged image panels, corruption chips, stacked routing bar, telemetry tiles, JSON block).
**The Stitch mock-ups contain placeholder content invented by Stitch** (face photos in Task 1, "RTX 4090",
"TensorRT", "CycleGAN", 512/1024 px, FID/PSNR values, "Download SVG", stroke-density slider, …).
Only the visual design was adopted; the implemented app shows real backend values only, and the report
says so explicitly.

## 1. What to generate in Stitch

Create one project, "Restoration & Sketch Studio", with these screens (desktop + mobile):

1. **App shell**: left sidebar (collapses to a top bar on mobile) with 4 workspaces + "System status"; backend status indicator.
2. **Universal Restoration** (Task 1): input card (Upload / Clean sample tabs, drag-and-drop), corruption card (type segmented control, severity Low/Medium/High/Custom, sliders, seed field), primary "Run restoration" button, results card (Clean reference / Model input / Restored output panels), system-information card (inference time, preprocess, total, PSNR/SSIM, corruption settings JSON).
3. **Hard-Routed Restoration** (Task 2): same layout + "Classifier & routing" card with 4 probability bars, predicted corruption, selected expert, classifier/expert time.
4. **Soft Mixture-of-Experts Restoration** (Task 3): same layout + "Routing weights" card with 4 weight bars, stacked contribution bar, strongest/second contributor, routing entropy.
5. **Face-to-Sketch Generator** (Task 4): Upload / Webcam tabs, Style 1/2/3 selector, "Generate sketch" button, side-by-side Original / Generated sketch panels, Download PNG button.
6. **System status**: backend info and a list of the 7 ONNX models with available/loaded/missing badges.

## 2. Prompt to paste into Google Stitch

```
Design a clean, modern web application called "Restoration & Sketch Studio" for a university
Generative AI project. Light theme, white cards with subtle borders and rounded-2xl corners on a
slate-50 background, indigo (#4f46e5) as the accent colour, Inter font, generous spacing.

Layout: a fixed left sidebar (on mobile a horizontal top navigation) listing five pages:
"Task 1 · Universal Restoration", "Task 2 · Hard-Routed Restoration",
"Task 3 · Soft Mixture-of-Experts Restoration", "Task 4 · Face-to-Sketch Generator",
"System status". At the bottom of the sidebar show a small green/amber/red dot with
"Backend ready".

Restoration pages (Tasks 1-3) use a two-column layout: a narrow left column of stacked cards
and a wide right column for results.
Left column: (1) "Input image" card with segmented tabs Upload | Clean sample, a dashed
drag-and-drop area, and a 4-column grid of sample thumbnails; (2) "Corruption" card with a
segmented control None | Salt & pepper | Gaussian blur | Occlusion, a severity segmented
control Low | Medium | High | Custom, sliders for custom parameters and a seed input;
(3) a full-width primary button "Run restoration".
Right column: a "Result" card with three square image panels side by side labelled
"Clean reference", "Model input", "Restored output" (download button under the output), and
below it two cards: a task-specific analysis card and a "System information" card with four
small stat tiles (Inference ms, Pre-process ms, Corruption ms, Total ms), two tiles for
PSNR and SSIM "input -> output", and a dark code block showing the corruption settings.
Task 2 analysis card: four horizontal probability bars (Clean, Salt-and-pepper, Gaussian
blur, Occlusion) with percentages, the top one tagged "top", plus tiles "Predicted
corruption" and "Selected expert".
Task 3 analysis card: four horizontal routing-weight bars (Identity, Salt expert, Blur
expert, Occlusion expert), a stacked colour bar of all four weights, and tiles "Strongest
contributor", "Second", "Routing entropy".
Face-to-Sketch page: left cards "Face photograph" (Upload | Webcam tabs with a camera preview
and "Capture photo" button) and "Sketch style" (Style 1 | Style 2 | Style 3), button
"Generate sketch"; right card shows "Original photograph" and "Generated sketch" side by side
with a "Download PNG" button.
System status page: a backend info card and a list of 7 ONNX models with status badges.
Use consistent colours for the corruption classes: clean/identity emerald, salt amber,
blur sky blue, occlusion rose. Make every page responsive.
```

## 3. Evidence to collect for the report

Save into `report/stitch/` (the LaTeX report already references these file names):

| File | Content |
|---|---|
| `stitch_overview.png` | Stitch project board showing all screens |
| `stitch_universal.png` | Stitch design of the Universal Restoration screen |
| `stitch_hard_routing.png` | Stitch design of Hard-Routed Restoration |
| `stitch_soft_moe.png` | Stitch design of Soft MoE Restoration |
| `stitch_face_to_sketch.png` | Stitch design of Face-to-Sketch |
| `stitch_mobile.png` | a mobile layout from Stitch |
| `stitch_prompt.png` | screenshot of the prompt / Stitch history (shows it originated in Stitch) |

Also export Stitch's generated HTML/CSS (if you use it) to `report/stitch/export/`, and note in
the report which design tokens (colours, radii, spacing) were transferred to
`frontend/src/index.css` (`@theme` block) and the components.

## 4. Aligning the implementation with the design

- Colours/fonts: edit the `@theme` tokens in `frontend/src/index.css`.
- Card/panel/button styles: `frontend/src/components/ui.jsx`.
- Page layout: `frontend/src/workspaces/RestorationWorkspace.jsx`, `FaceToSketch.jsx`, `App.jsx`.
- Then take application screenshots (`report/figures/app_*.png`) for the report.
