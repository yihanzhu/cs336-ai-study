# Evidence scope and missing originals

Version 3.1 · 2026-10-22. The teaching layer has 299 English and 299 Chinese plans with shared IDs and source identities. This means complete plans for the defined curriculum; it does not mean every original classroom detail was recovered.

| Material | Evidence baseline 1.2 |
|---|---|
| Course schedules | 57 entries across 2024–2026, mapped to 29 teaching chapters |
| Core sources and assignments | 51 core files and 17 assignment PDFs |
| Captions | 34 uploaded tracks plus local ASR for 2026 L08; 56527 segments, 35 chronological text reviews |
| Original audio | 35 recordings; machine comparison plus 267 flagged-region textual checks; not full manual listening |
| PDFs | 42 files, 1889 page layouts/main figures viewed; not every tiny label or original experiment independently checked |
| Images and sampled video | 231 unique lecture images; 266 GIF frames; 158 sampled video times, not all frames |
| Publisher execution records | 19 JSON traces, 2 stdout files; 405 compound outputs inspected, selected relationships recomputed; not local GPU replay |
| Teaching experiments | 10 executed CPU reconstructions; not large-model/GPU performance reproduction |

## Explicit gaps

- No confirmed official 2024 video playlist was obtained. Official 2024 L17 page 2 says the upcoming Ce Zhang and Aakanksha Chowdhery guest classes were not recorded; their standalone class slides were not obtained.
- Full 2025 Junyang Lin and Mike Lewis and 2026 Daniel Selsam classes were not obtained. Qwen3/Llama3 publications are related reading, not recovered classroom Q&A or demos.
- Dan Fu 2026 has class video/captions plus related Parcae/Megakernel sources; standalone slides for that class were not obtained. GPU execution was not reproduced.
- Official 2026 L08 captions remain missing. Local dual-ASR and source comparisons help, but some distant student questions cannot be recovered. ASR systems can share errors; guesses are not quoted as original speech.
- Video inspection is sampled. Cropped screens and intervening changes remain unverified. All-page layout inspection does not independently establish every footnote or experimental datapoint.
- 2026 L16 did not fully develop a complete long-context extension recipe. No complete verified recipe is invented.
- Current model ranks, prices, legal conclusions, hardware specifications, and software support require primary-source checks when taught. This package is not an exhaustive rolling survey after spring 2026.

## Bilingual checks

Point IDs, prerequisite IDs, next-point IDs, source URLs and hashes, extension coverage, and all six teaching fields are checked across languages. Examples, questions, judgment criteria, formulas, and boundaries were reviewed by the same author; this is not independent external peer review. Numeric-text differences include translated number words, units, and simplified wording; candidate reviews are recorded in CS336-bilingual-review.json. Mechanical counts and hashes do not prove semantic correctness.

The main teaching layer and point-level limits are bilingual. Historical detailed preparation/audit narratives remain in Chinese; retained JSON and original primary-source links provide the evidence trail. Full raw captions, PDFs, audio/video, local learner state and recovery caches are excluded from Git and the share package.
