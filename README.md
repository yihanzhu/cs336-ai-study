# CS336 AI tutoring · English / 中文

[中文说明](README.zh-CN.md) · [Start tutoring](outputs/START-HERE.txt) · [Bilingual reader](outputs/CS336-lesson-plans.html)

299 English lesson plans and 299 Chinese counterparts share point IDs and primary-source identities. The reader defaults to English; teaching follows each learner's chosen language. An AI explains one point, works an example, asks one question, and waits. No prerequisite videos or large assignments.

## Start with any agent

1. Clone or download this curriculum and open it in an agent that can read files.
2. Create private local progress from a blank template:

   ```sh
   python3 work/init_learning.py --language en
   # Or: python3 work/init_learning.py --language zh
   ```

3. Tell the agent: “Read AGENTS.md, outputs/START-HERE.txt, and my local/learning-progress.md. Continue in my saved language. Ask one understanding question, then wait.”

Without Python, create `local/` and copy the desired blank template there as `learning-progress.md`. Never overwrite existing progress. Without file tools, provide the lesson and progress text to the agent and save its updated progress yourself. Codex uses AGENTS.md; CLAUDE.md imports the same rules for compatible Claude Code sessions. Other agents can read these instructions explicitly.

## Personal progress stays local

| Public curriculum | Private learner state |
|---|---|
| Lessons, sources, blank progress templates | `local/learning-progress.md` |
| Shared through Git and releases | Ignored by Git; excluded from releases |
| Curriculum updates can be pulled | Never automatically committed or uploaded |

A new clone contains no personal history. To continue on another computer, separately carry your local progress file, or explicitly choose your own private storage. The curriculum repository provides no learner accounts, telemetry, progress server, or cloud-sync setup. Agent/provider chat retention and remote processing depend on the tool you choose; local progress does not imply local model execution.

## Files

| File | Purpose |
|---|---|
| `outputs/CS336-lesson-plans.html` | Self-contained English/Chinese reader |
| `outputs/CS336-lesson-plans.md` | English plans and tutor feedback |
| `outputs/CS336-逐点陪学教案.md` | Complete Chinese counterpart |
| `outputs/CS336-bilingual-lessons.json` | Both languages and source links |
| `outputs/CS336-curriculum.en.md` | Curriculum and chapter-level sources |
| `outputs/CS336-evidence-and-gaps.en.md` | Evidence scope and missing originals |
| `outputs/CS336-bilingual-review.json` | Author parity-review accounting |
| `outputs/CONTINUE-LEARNING.md` | Local progress and cross-device handoff |

## Evidence and licenses

Complete means plans for the defined 299-point curriculum, not recovery of every original classroom detail. Full materials for five guest classes were not obtained; Dan Fu has video/captions but no standalone class slides. The 1.2 evidence baseline has explicit audio, sampled-video, PDF and execution-review limits. No full manual listening, all-frame inspection, GPU-training reproduction, or independent full-course peer review is claimed.

The teaching layer and point-level source gaps are bilingual. Historical detailed preparation/audit notes remain in Chinese. Counts, hashes and structural tests do not prove every semantic claim.

MIT applies to this project's original code and teaching text; see LICENSE and THIRD_PARTY_NOTICES.md. Linked Stanford materials, papers, datasets, model code and recordings retain their own terms. Their original PDFs, full captions, media, frames and execution caches are not redistributed. This is an independent study project, not an official Stanford course, translation or certification.

## Build and check

Tracked files suffice to rebuild teaching outputs without raw-source caches or an existing ZIP:

```sh
python3 work/build_bilingual.py
python3 work/verify_lesson_plans.py
python3 work/verify_bilingual.py
python3 work/verify_publication.py
```

The full evidence rebuild (`python3 work/build_release.py`) needs local source caches. Standard-library teaching experiments run with `python3 outputs/CS336-教学小实验.py`.
