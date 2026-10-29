# Actual editor replay

`../demo.mp4` is a15-second,1440×960,24fps captioned replay of five actual browser captures from the frozen source identified in `../evidence/editor-final-source.json`. Each stage holds for3seconds. The pauses are edited; this is not a real-time performance recording. No cached-picture interpolation, synthetic cursor or fabricated field is used. There is no audio.

1. Initial heat computation accepted after primal and adjoint checks.
2. Local contrast+5% changes the operator and recomputes a new checked solution.
3. Mesh20→32 invalidates the old mapping.
4. A one-iteration solve fails. The old image is visibly stale; no current gradient/export is available.
5. A fresh sparse-direct fallback passes the original-equation checks.

Raw JPEGs, accessibility observations and media properties remain beside this file. The source hashes/manifest and independent review are in `../evidence`. The actual runtime/assembly source is frozen across these final captures. Earlier separately named mobile observations are not relabeled as final-source captures.

Rebuild from repository root with `python scripts/make_demo.py` after installing the study extra and free FFmpeg. The script uses Matplotlib's installed DejaVu fonts without redistributing font files, renders into temporary clips, and replaces the previous master only after a complete successful render. Samples from the local-edit and rejected stages were visually inspected; duration/frame count were checked with FFprobe. Full audiovisual perception is limited to these static cuts and their explicit holds, with no motion or sound to assess.
