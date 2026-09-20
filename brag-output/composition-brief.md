# Hyperframes Composition Brief: open-jev

> **Status: NOT YET COMPOSED OR RENDERED.** Steps 1-2 of `/brag` are complete
> (`brag-output/brag-plan.md`). Steps 3-4 were deliberately deferred: this machine hosts a
> production vLLM server (~84 GB) plus a training queue with ~5 GB RAM free, and a headless
> browser render could OOM the host. Run steps 3-4 when the queue is idle. This brief is written
> to be executable by an agent with no memory of the planning session.

## Objective
Create a short launch-style brag video for open-jev.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`, poster `brag-output/brag.jpg` baked as frame 0
- Share copy: `brag-output/share-copy.txt`
- Format: **landscape — 1920x1080**
- Duration: **24.0s** (scene budget in the plan sums to 24.02s)

## Source Material
- Project root: `/home/spark_4551/projects/open-jev`
- Primary files read: `README.md` (Quickstart + Results table + Findings), `pyproject.toml`,
  `openjev/views.py`, `openjev/serve.py`, `openjev/cli.py`, `openjev/check.py`,
  `tasks/agnews.yaml`, `tasks/banking77.yaml`, `results/agnews.json`,
  `results/agnews-nogold.json`, `results/banking77.json`, `docs/task-spec.md`
- Product name: `open-jev`
- Tagline: `Distill a prompt into a small, fast, calibrated classifier.`
  (first clause of `pyproject.toml → description`)
- Key visual moment to recreate: the `curl` → typed-JSON round trip from the README Quickstart,
  inside a dark terminal card
- **There is no website, no `index.html`, no CSS, and no logo in this repo.** Do not go looking
  for one, and do not invent a brand mark. The terminal and the JSON are the product's face.

### Copy that must appear verbatim
1. `On the hard Russian tasks the framework did not deliver.` — README, Findings (Scene 1)
2. `Distill a prompt into a small, fast, calibrated classifier.` — `pyproject.toml` (Scene 2)
3. `"choice": "Business"` and `"confidence": 0.987` — README Quickstart response (Scene 3)
4. `agnews    student 0.879    teacher 0.880` — README Results table (Scene 4)
5. `0.8825 with zero gold labels anywhere` / `teacher 0.8815 on the same rows` — README Findings,
   "A task with no gold at all gets the same student" (Scene 4)
6. `77 options, 19 at a time — top_logprobs caps at 20` — README "How it works" §2 + Limitations
7. `The teacher is the ceiling.` — README, Findings (Scene 6)

### Numbers: traceability contract (READ THIS BEFORE WRITING ANY TEXT)
This project's entire voice is "only numbers this repo produced." **Do not round, restate, blend
or improve any number.** Every figure permitted on screen, with its source:

| On screen | Exact source value | Where |
|---|---|---|
| `0.879` student / `0.880` teacher | 0.8794999718666077 / 0.8799999952316288 | `results/agnews.json`, README table row `agnews` |
| `0.8825` / `0.8815` | gold_acc_offline / teacher_gold_acc_offline | `results/agnews-nogold.json` |
| `0.987` (three times: Business, confidence) | 0.9872164130210876 | README Quickstart response |
| `0.006 / 0.005 / 0.002` | 0.005825810134410858 / 0.005110130645334721 / 0.0018476743716746569 | README Quickstart response |
| `5.3 ms` batch 1 | gpu_b1_p50 = 5.328 | `results/agnews.json` |
| `869 ex/s` at batch 64 | throughput_gpu_b64 = 868.59 | `results/agnews.json` |
| `2644 ex/s` | throughput_gpu_b64 = 2644.01 | `results/banking77.json` — **this is banking77, NOT agnews** |
| `77 options` / `6 teacher calls per example` | k=77, 33000 calls / 5500 rows | `tasks/banking77.yaml`, `results/banking77.json` |
| `top_logprobs caps at 20` | — | README "How it works" §2 |

Rounding the four probabilities to 3dp in Scene 3 is the **only** permitted simplification, and
it is a legibility decision, not a claim. If the JSON block can be made legible at full precision,
prefer full precision.

**Never** pair `2644 ex/s` with agnews, never write "2644 decisions/s" next to the 0.879/0.880
parity row, and never state a latency without `batch 1`. Those are the exact conflations this
project's README exists to prevent.

## Creative Direction
- Tone preset: **deadpan**
- Creative direction: *a benchmark table that refuses to oversell itself*
- Interpretation: long holds, one thought per scene, large sparse monospace on near-black, a
  single accent colour on the number that matters. Hard cuts, unhurried. Nothing bounces, pulses,
  glows or celebrates. Motion exists only to show a value arriving.
- Angle: every launch video opens with its best number; this one opens with its worst, verbatim
  from the README, and earns the right to show the good ones. open-jev's credibility *is* the
  feature — the video is structured like the repo: failure, measurement, caveat.
- Hook (first 2-3s): the amber line `On the hard Russian tasks the framework did not deliver.`
  on black, attributed to the README, with no music accent. (Alternative hook B and the scene
  surgery it requires are at the bottom of `brag-plan.md` — use it only if the user picked it.)
- Outro / punchline: the name, `MIT · one YAML · runs next to your teacher on one box`, then the
  callback footnote `The teacher is the ceiling.` holding alone as the music fades.
- Avoid:
  - Generic SaaS language ("streamline", "supercharge", "10x", "effortless") — banned outright
  - Any superlative the README does not make. The README makes almost none.
  - Abstract filler visuals: particle fields, gradient meshes, glowing orbs, neural-net graphics,
    fake dashboards, stock "AI" imagery
  - Charts. The plan uses flat number rows on purpose; a bar chart would oversell a 0.001 gap.
  - Any mention or implied comparison of TypeSafe's Jev performance — the README explicitly
    refuses to repeat unverifiable closed-API claims. Naming Jev at all is out of scope here.

## Visual Identity
No stylesheet exists in the repo; these values are **derived** from the product's real surface
(dark terminal, JSON, the repo's own `WARNING` lines), not extracted from a file.
- Background: `#0B0D10`
- Surface / terminal chrome: `#14181D`, border 1px `#232A31`, radius 8px
- Text: `#E6E9EC`
- Muted (JSON keys, captions, attributions): `#7C8794`
- Accent — the number that matters: `#7CD4A0`
- Warning accent — the honest-failure line and the `top_logprobs` constraint: `#E0A458`
- Display font: JetBrains Mono 600
- Body font: JetBrains Mono 400
  (fallback chain `ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace`; if the font
  cannot be bundled deterministically, use the fallback rather than a proportional face —
  monospace everywhere is the identity, do not substitute Inter/Helvetica anywhere)
- Visual references from the project: the six-line JSON response; the `$ curl` prompt line; the
  77-row option list in `tasks/banking77.yaml`; the README results table's flat two-column feel

## Storyboard
`brag-output/brag-plan.md` is the creative contract — read its Storyboard section in full before
writing HTML. Scene summary:

1. **The failure line** — 3.82s (0.00→3.82) — one amber sentence on black + README attribution.
   Main line settled ≥3.0s. No SFX, near-silent bed.
2. **What it is** — 3.82s (3.82→7.64) — `open-jev` + the pyproject tagline. Tagline settled ≥2.7s.
3. **The typed decision** — 5.47s (7.64→13.11) — **centrepiece.** Terminal card; `$ curl …` types
   out with key ticks; the JSON response lands as one unit on the **8.74s strong cue** with a soft
   impact; `"Business"` + `0.987` tint to `#7CD4A0` at 10.93s; latency caption at 12.02s. The JSON
   holds ≥4.4s — the longest hold in the video, by design.
4. **Two numbers** — 4.36s (13.11→17.47) — row 1 (parity) at 13.11, row 2 (zero gold) at 15.29.
   2.18s apart, **not** consecutive beats. Both hold to the cut. Identical quiet tick per row.
5. **77 classes** — 2.72s (17.47→20.19) — 77 real banking77 option chips collapse to 19 at the
   17.47s strong cue; caption at 18.02s settled ≥1.8s.
6. **It ends on the caveat** — 3.83s (20.19→24.02) — name + MIT line at 20.75s, footnote callback
   at 21.84s settled ≥2.2s, alone as the music fades. No fade-to-black wipe.

Readability floors are load-bearing, not advisory: short label ≈0.8s settled, sentence ≈0.3s per
word (min 1.2s). Scene 1's hook line and Scene 4's row 2 are the two tightest — if implementation
pushes either below its floor, **lengthen the scene, do not speed up the text.**

## Audio
- Audio role: **sparse professional accents over a low bed.** Four SFX cues in 24 seconds, total.
- Audio arc: near-silence under the failure line → bed becomes audible with the product name →
  one physical impact as the JSON lands (8.74s) → two flat identical ticks for the two numbers →
  one low impact on the closing title → silence under the final caveat.
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (bundled with the
  brag skill). Chosen because its first strong cue is at 8.74s, so the opening is sparse — the
  other four bundled tracks all hit inside the first 4 seconds and would trample the hook.
- Music treatment: start 0.0s; gain ~0.28 for Scenes 1-2, lift to ~0.42 at 8.74s, fade
  22.40s → 24.02s to zero. The track is cheerful by default: if it fights the writing, apply a
  high-shelf cut rather than swapping the track (the cue preset is what makes it work here).
- Music cue guidance: preset at
  `<brag-skill>/assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.md`
  (109.96 BPM). Use **three** strong-cue locks only: `8.74s` (JSON arrival), `13.11s` (first
  number row), `17.47s` (chip collapse). `22.93s` is available but should be left unused — the
  footnote is stronger with nothing under it. Beat grid for entrances:
  0.56 / 1.09 / 1.64 / 2.19 / 2.73 / 3.27 / 3.82 / 4.39 / 4.91 / 6.00 / 6.56 / 7.09 / 7.64 /
  8.19 / 8.74 / 9.29 / 9.83 / 10.37 / 10.93 / 11.46 / 12.02 / 12.55 / 13.11 / 13.64 / 14.20 /
  14.73 / 15.29 / 15.84 / 16.38 / 16.93 / 17.47 / 18.02 / 18.56 / 19.10 / 19.66 / 20.19 / 20.75 /
  21.28 / 21.84 / 22.37 / 22.93 / 23.46 / 24.02.
  For anything with words in it use **every other** beat; consecutive beats are for non-text
  elements (probability rows, chip dropout) only.
- Audio-reactive treatment: **none.** Music-energy-driven glow/breathing reads as hype and
  contradicts the tone. Do not add it even if the runtime makes it cheap.
- Audio-coupled moments:
  - Scene 3, `$ curl …` line — typing animation with sparse key ticks (not one per character)
  - Scene 3, 8.74s — the JSON block lands: one soft plate/tin impact, dry, no tail
  - Scene 4, 13.11s and 15.29s — one quiet tick each, **same file, same gain** (the sameness is
    the deadpan)
  - Scene 6, 20.75s — one dry low impact under the title
  - Scene 1 and the Scene 6 footnote — **silence, deliberately.** Do not fill them.
- SFX selection guidance: read `<brag-skill>/assets/sfx/sfx-analysis.md` and prefer low
  high-frequency-risk samples; `sfx/keyboard` for the typed line, `sfx/impact` (soft plate/tin,
  not punch) for the JSON arrival, `sfx/ui` or `sfx/interface` for the two ticks. No risers, no
  whooshes, no swells, no stingers, no success chimes.
- Exact SFX choice: Hyperframes picks filenames, timestamps and gains after the animation exists.
- Audio files: copy the chosen music and SFX into `brag-output/composition/assets/`.

## Hyperframes Instructions
Load `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`, `hyperframes-keyframes`,
`hyperframes-cli`. `/brag` is its own workflow: do **not** enter the `hyperframes` entry-point
intent interview and do not route into its generic promo / launch-video workflow. Prefer native
Hyperframes conventions over anything in this brief where they conflict on *mechanism*; this brief
wins on *content, copy, numbers and tone*.

Requirements:
- Scene 3 must show real product output (it does — verbatim README Quickstart request/response).
- Every text element must clear its readability floor in the final render.
- Keep the video within 15-25s.
- Include the planned music + 4 SFX cues.
- Treat audio notes as guidance; choose SFX after the animation exists.
- Treat cue metadata as optional hints; 3 strong-cue locks maximum.
- Honour the 22.40→24.02s music fade and the silent final footnote.

### Deferred-render notes specific to this machine
- **Do not run any browser, `npx`, install, render, or teacher call until the user confirms the
  training queue is idle.** The host had ~5 GB RAM free at plan time with a production vLLM
  resident; a headless Chrome render can OOM it and kill the server.
- Three other agents were working in this repo at plan time. Write only inside `brag-output/`.
  Do not edit `README.md`, `results/*`, `tasks/*` or anything under `openjev/`.
- Do not `git commit`.

### Assets to capture before composing (none exist yet)
All of these are currently **text I transcribed from source files**, not captured artefacts. When
the render window opens, decide per item whether to recreate it in HTML (preferred — it stays
crisp at 1080p and is themeable) or to capture it for reference:
1. **The `curl` request line** — README Quickstart lines 47-48. Recreate in HTML.
2. **The typed JSON response** — README Quickstart lines 54-56 (full precision available there).
   Recreate in HTML, pretty-printed. This is the single most important asset in the video.
3. **The agnews results row** — README Results table row `agnews`; exact values in
   `results/agnews.json`. Recreate as two flat text rows.
4. **The zero-gold numbers** — `gold_acc_offline` / `teacher_gold_acc_offline` in
   `results/agnews-nogold.json`. Text only.
5. **The 77 option strings** — `tasks/banking77.yaml`, `options:` list, in file order. Needed
   verbatim for the Scene 5 chip grid; the first 19 in file order are the natural "shortlist" row.
6. **Latency / throughput caption** — `results/agnews.json` (`latency_ms.gpu_b1_p50`,
   `throughput_gpu_b64`) and `results/banking77.json` for the 2644 figure only.
7. *(Optional, only if a live terminal frame is wanted)* a real `openjev check tasks/agnews.yaml`
   run — **it makes no teacher calls** and is the one safe live capture. `--probe` is NOT safe:
   it calls the teacher on :8000. `openjev serve` + `curl` would also be real but loads a model
   and is not worth the RAM on this host.
   **Nothing in this list requires the teacher, a GPU, or a model load.**

## Self-review checklist before render
- [ ] Every number on screen appears in the traceability table above, unchanged.
- [ ] `2644 ex/s` appears only next to banking77, if at all.
- [ ] No latency figure appears without `batch 1`.
- [ ] Scene 1's hook line is settled ≥3.0s; Scene 4 row 2 ≥1.8s; Scene 5 caption ≥1.8s.
- [ ] Scene durations sum to 15-25s.
- [ ] Nothing glows, pulses, bounces or reacts to music energy.
- [ ] No word appears on screen that the repo does not use about itself.
- [ ] The video ends on `The teacher is the ceiling.` with no sound under it.
- [ ] `npx hyperframes check` passes with zero errors in `brag-output/composition/`.
