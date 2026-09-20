# Brag Plan: open-jev

Steps 1 and 2 only. Step 3 (Hyperframes composition) and step 4 (render) are deliberately
not run — see `composition-brief.md` for the handoff.

---

## Step 1 rubric (answered)

**1. What is the app?**
`open-jev` distills a local LLM teacher (Qwen3.8-27B on vLLM) into a ~140M encoder that returns a
*typed* decision — `choice` / `score` / `noul` — with calibrated probabilities at
`POST /v1/systemone`, instead of a string you have to parse. An open reimplementation of the idea
behind TypeSafe's closed Jev API, with only numbers the repo's own code produced.

**2. What is the most impressive / most characteristic claim?**
Not a claim — a refusal. The README's Findings section opens beats with what *didn't* work:
> "On the hard Russian tasks the framework did not deliver."
Runner-up, and the strongest real result: an agnews run with **no gold labels anywhere** scores
**0.8825** against the real ag_news test labels, vs the teacher's **0.8815** on the same 2000 rows.

**3. What is the visual hook?**
The typed JSON response. Four label keys, four probabilities, one `confidence` — the product's
entire value proposition fits in six lines of monospace. Second visual: the 77-option `banking77`
label list collapsing into a 19-option shortlist.

**4. What should be shown from the actual UI?**
There is no UI and no website. The product's face is (a) the `curl` → JSON round trip in the
README Quickstart, (b) the results table, (c) `openjev check --probe` warning output. Scenes are
recreations of real terminal output, not mockups of an imaginary dashboard.

**5. What is the shortest satisfying video?**
~24s. Below ~20s the honest-failure hook has no room to be paid off, and the JSON needs a real
hold to be read rather than glanced at.

**6. What tone fits best?**
Preset: **`deadpan`**. Creative direction: *"a benchmark table that refuses to oversell itself."*
No `--tone` flag was given. The audience is ML engineers, and the repo's own voice is
measurement-first and anti-hype: the README's Findings section leads with the failure, states the
teacher as a ceiling, and calls its own augment result "a small gain, not a result." `chaotic`,
`cinematic` and `app-store` would all contradict the source text. `polished` was the other
candidate and was rejected because polished shows only wins — `deadpan` is the only preset that
can put the failure line on screen as the *hook* and have it read as confidence.

**7. What should the audio feel like?**
Low, sparse, restrained. Music on (no `--no-music`), no narration. Music is a bed, not a driver;
SFX are dry mechanical ticks (keys, one plate impact) matched to real motion. Nothing celebratory.

**8. What should the share caption say?**
"A 140M encoder that keeps its teacher's accuracy at 5.3 ms a decision — and a README that tells
you exactly where it doesn't. open-jev, MIT."

**9. What's the user flow worth showing?**
Three beats, straight from the Quickstart:
`openjev run tasks/agnews.yaml` (label → train → calibrate → eval) → `openjev serve runs/agnews`
→ `curl /v1/systemone` returns a typed JSON decision. The video shows beat 3 in full (the payoff)
with beats 1–2 compressed into one command line, because the JSON *is* the product.

---

## What is this app?
A framework that turns one YAML task file into a small, fast, calibrated classifier by distilling
a local LLM teacher's soft labels into a ~140M encoder you can serve yourself.

## The angle
Every launch video opens with its best number. This one opens with its worst, verbatim from the
README, and then earns the right to show the good numbers. The premise: **open-jev's credibility
is the feature.** The video is structured like the repo — failure first, then the measurement,
then the caveat again at the end. A viewer should come away thinking "these numbers are probably
real," which is the only thing that matters to an ML engineer deciding whether to clone it.

## Hook (first 2-3 seconds) — CHOSEN
Black screen, monospace, one line, no motion but the cut:

> **On the hard Russian tasks the framework did not deliver.**

Attribution fades in underneath at 1.6s: `— README.md, "Findings"`. No music swell, no logo.
The tension is entirely "why is a launch video saying this?"

*(Alternative hook B is at the bottom of this plan.)*

## Key moments (the middle)
- The typed JSON response arriving key by key — `"choice": "Business"` and `"confidence": 0.987`
  landing last and holding, with `5.3 ms · batch 1 · GB10` set small beneath it.
- Two numbers, stated flat, no chart: `student 0.879 / teacher 0.880` on agnews, then
  `0.8825 with zero gold labels anywhere`.
- 77 `banking77` option chips collapsing into a 19-chip shortlist, captioned with the real reason:
  `top_logprobs caps at 20`. The constraint is the content.

## Outro / punchline
The name and licence, then the hook line returns as a footnote — smaller, still true, never
walked back: `The teacher is the ceiling.` The video ends on its own caveat. That is the joke,
and it is also the pitch.

## User flow worth showing
Entry → key action → result: `openjev run tasks/agnews.yaml` → `openjev serve runs/agnews` →
`curl /v1/systemone` → a typed JSON decision with probabilities. Scene 3 is the centrepiece and
must be a real terminal recreation using the exact Quickstart input string and the exact response
from the README.

## Tone
- Preset: **deadpan**
- Creative direction: *a benchmark table that refuses to oversell itself*
- Interpretation: long holds, one thought per scene, large sparse monospace on near-black, no
  colour except one accent on the number that matters. Cuts are hard but unhurried; nothing
  bounces, nothing pulses, nothing celebrates. Motion exists only to show a value arriving.

## Format: landscape — 1920x1080
## Duration: 24.0s (six scenes; deadpan normally runs 3-4, the two extra exist because
"show the thing" needs the terminal *and* the JSON, and the banking77 constraint is a visual, not
a text read)

## Visual identity (from the project)
No CSS, HTML or brand assets exist in this repo — it is a CLI and a Python package. The palette
below is **derived** from the product's actual surface (a dark terminal, JSON, and the repo's own
`WARNING` lines), not extracted from a stylesheet. Stated explicitly so nobody later believes
these values came from a file.

- Background: `#0B0D10` (near-black terminal)
- Surface / terminal chrome: `#14181D`, 1px border `#232A31`
- Text: `#E6E9EC`
- Muted / keys / captions: `#7C8794`
- Accent (the number that matters — probabilities, the winning value): `#7CD4A0`
- Warning accent (the honest-failure line, the `top_logprobs` constraint): `#E0A458` — grounded in
  the repo's real `WARNING the teacher under-predicts ...` output
- Display font: JetBrains Mono (600) — the project has no other face; monospace *is* the identity
- Body font: JetBrains Mono (400)
- Strongest visual element: the six-line typed JSON response from the README Quickstart

## Share copy (draft)
A 140M encoder that keeps its teacher's accuracy at 5.3 ms a decision — and a README that tells
you exactly where it doesn't. open-jev: one YAML, a local teacher, a typed decision you can
threshold on. MIT.

## Audio direction
- Role: sparse professional accents over a low bed. The bed is almost subliminal for the first
  8 seconds so the hook lands in near-silence.
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (bundled, 109.96 BPM). Chosen
  because its first strong cue is at **8.74s** — the opening is sparse, which is exactly what a
  deadpan hook needs. The other four bundled tracks all hit hard inside the first 4 seconds.
- Music treatment: start at 0.0s, gain ~0.28 for scenes 1-2, lift to ~0.42 from 8.74s, fade out
  22.4s → 24.0s. Low-pass or shelf the top end if it reads as too upbeat — the track is cheerful
  by default and must not argue with the writing.
- Music cue guidance (preset read from `assets/music/cues/...vol-12...music-cues.md`):
  - `8.74s` (1.00) — the JSON block arrives. The single biggest reveal in the video.
  - `13.11s` (0.98) — the first number row.
  - `17.47s` (0.99) — the 77 chips collapse.
  - `22.93s` — hold; the footnote is already on screen, no accent needed.
  - Beat grid for sequential reveals: `8.74 / 9.29 / 9.83 / 10.37 / 10.93` — use **every other**
    beat for anything with words in it; consecutive beats only for the non-text probability rows.
  - Restraint note: deadpan. Beat-align entrances, never beat-align *emphasis*. No pumping.
- Audio-reactive treatment: **none.** Any breathing/glow driven by music energy would read as hype
  and contradict the tone.
- SFX posture: sparse. Four cues total, all motion-matched: key ticks under the typed command, one
  soft plate/tin impact as the JSON block lands, one quieter tick per number row, one dry low
  impact on the final title. Nothing on the hook — the hook plays dry.
- Audio-coupled moments: the typed command line (key ticks); the JSON block arrival (impact on
  8.74s); the two number rows (ticks); the 77→19 chip collapse (one soft sweep or nothing).
- Restraint rule: no riser, no whoosh, no swell, no stinger on the failure line, no sound on the
  final footnote. If a cue makes a moment feel *exciting* rather than *stated*, cut it.

---

## Storyboard

Total: **24.0s**. Cut points are placed on the vol-12 beat grid. Every text element's settled
(fully entered, not yet exiting) time is listed and clears the readability floor
(short label ≈0.8s; sentence ≈0.3s/word, min 1.2s).

### Scene 1 — "The line the README opens with" — 3.82s (0.00 → 3.82)
Full-bleed `#0B0D10`. Nothing else on screen. At 0.27s one line of JetBrains Mono 600, ~64px,
centre-left, in `#E0A458`, slams in over 0.2s (opacity + 8px rise, no scale, no blur):

> `On the hard Russian tasks the framework did not deliver.`

At 1.64s, small `#7C8794` 20px underneath: `— README.md · "Findings"`.
Settled read: main line **3.35s** (floor 3.0s ✓), attribution 2.1s (secondary).
Sequential/interaction: none. One line, one hold. The stillness is the point.
Audio intent: near-silence. Music bed barely present. **No SFX.**
Audio-coupled idea: none — deliberately dry.
Music: low bed, ~0.28 gain, no cue lands here (first strong cue is 8.74s).
Transition mood: hard cut, no flash → Scene 2

### Scene 2 — "What it is" — 3.82s (3.82 → 7.64)
Hard cut. Same background. `open-jev` in `#E6E9EC` 600 at ~88px, left-aligned on a 12-column
grid, enters at 3.82s (0.25s, opacity + 6px rise). At 4.39s, beneath it in `#7C8794` 28px:

> `Distill a prompt into a small, fast, calibrated classifier.`

(verbatim from `pyproject.toml → description`, first clause; the full sentence continues
"an LLM teacher's soft labels train a ~140M encoder you can serve" — do **not** try to fit both.)
Settled read: title 3.6s, tagline **2.9s** (floor 2.7s ✓).
Sequential/interaction: two-step reveal, title then tagline. Both hold to the cut.
Audio intent: the bed becomes audible. Still no accent.
Audio-coupled idea: none.
Music: bed, ~0.30.
Transition mood: hard cut → Scene 3

### Scene 3 — "Show the thing: the typed decision" — 5.47s (7.64 → 13.11) — CENTREPIECE
A terminal card, `#14181D` on `#0B0D10`, 1px `#232A31` border, rounded 8px, occupying the centre
~72% of frame. Three dots or a plain title bar reading `openjev serve runs/agnews · :8099`.

1. **7.64s** — a prompt line is already present, typed out over ~0.5s with key ticks:
   `$ curl -s localhost:8099/v1/systemone -d '{"model":"agnews","input":"Shares of the airline fell 8% …"}'`
   Wrap or truncate the input string with an ellipsis — it is texture, not a required read.
   The input sentence is the real one from the README Quickstart.
2. **8.74s (strong cue)** — the response block lands as one unit (0.25s, opacity + 4px rise), soft
   plate impact. Pretty-printed, keys `#7C8794`, string values `#E6E9EC`:
   ```json
   {
     "model": "agnews",
     "choice": "Business",
     "probabilities": {
       "World": 0.006, "Sports": 0.005,
       "Business": 0.987, "Sci/Tech": 0.002
     },
     "confidence": 0.987
   }
   ```
   Values rounded to 3dp for legibility; the full-precision response is in the README and must be
   quoted exactly in the brief so the rounding is a deliberate, disclosed choice.
3. **10.93s (strong cue)** — `"Business"` and `0.987` tint to `#7CD4A0` and a hairline underline
   draws under the `Business` probability row. No zoom, no bounce.
4. **12.02s** — small `#7C8794` caption settles below the card, right-aligned:
   `5.3 ms · batch 1 · GB10 · 869 ex/s at batch 64`
   Holds to the cut (1.09s; short data label, acceptable as a glance-read, not a required read).

Settled read: JSON block on screen **4.4s** — the only element in the video that gets this long,
because it is the product.
Sequential/interaction: **yes** — command types out, response lands, the two values that matter
tint last. This is a demonstration, not a mockup.
Audio intent: the one moment the video allows itself a physical sound. Dry, single, no tail.
Audio-coupled idea: key ticks under the typed command (sparse, not every character); one soft
plate/tin impact exactly on 8.74s; nothing on the tint.
Music: bed lifts to ~0.42 at 8.74s.
Transition mood: hard cut → Scene 4

### Scene 4 — "Two numbers" — 4.36s (13.11 → 17.47)
Empty near-black. Two rows, left-aligned, monospace, generous leading. Nothing else.

- **13.11s (strong cue)** — row 1, 44px: `agnews    student 0.879    teacher 0.880`
  with `0.879` in `#7CD4A0`. Small muted label to its right: `parity, 2000 eval rows`.
- **15.29s** — row 2, 44px: `0.8825 with zero gold labels anywhere`
  `0.8825` in `#7CD4A0`. Muted second line, 20px: `teacher 0.8815 on the same rows`.

Both rows hold to the cut. Entrances 0.25s, opacity + 6px rise, no stagger inside a row.
Settled read: row 1 **4.1s** (floor 1.2s ✓), row 2 **1.9s** (floor 1.8s ✓).
Sequential/interaction: **yes** — two rows arriving one at a time, both held together afterwards.
Do **not** snap these to consecutive beats (0.55s apart at 110 BPM is unreadable); the 2.18s gap
is deliberate.
Audio intent: a flat statement of fact twice. Two identical quiet ticks, same sample, same gain —
the sameness is the deadpan.
Audio-coupled idea: one dry tick per row arrival, beat-aligned.
Music: bed steady ~0.42.
Transition mood: hard cut → Scene 5

### Scene 5 — "77 classes, and the reason it's hard" — 2.72s (17.47 → 20.19)
The one purely visual beat. A grid of 77 small muted chips (real `banking77` option strings from
`tasks/banking77.yaml` — `card arrival`, `pin blocked`, `top up failed`, `declined transfer`, …)
fills the frame at 17.47s, already present from the cut.

- **17.47s (strong cue)** — the grid collapses: 58 chips drop out (fade + 4px fall, staggered
  across 0.4s), 19 remain and settle into a single tidy row in `#E6E9EC`.
- **18.02s** — one line beneath, 32px, `#E0A458` on the numeral only:
  `77 options, 19 at a time — top_logprobs caps at 20`
  Settled **1.9s** (floor 1.8s ✓). Muted 20px under it, glance-read only:
  `6 teacher calls per example · 2644 ex/s served`

Sequential/interaction: **yes** — the collapse is the whole scene. It shows a real engineering
constraint being handled, not a feature being claimed.
Audio intent: the sound of something being narrowed down. One soft, short sweep or nothing at all
— if in doubt, nothing.
Audio-coupled idea: the chip dropout stagger may ride the 17.47 → 18.02 beat window.
Music: bed steady.
Transition mood: hard cut → Scene 6

### Scene 6 — "It ends on the caveat" — 3.83s (20.19 → 24.02)
Back to empty near-black, same composition as Scene 2 — a visual rhyme.

- **20.75s** — `open-jev` in `#E6E9EC` 600, ~88px.
- **20.75s** — beneath, `#7C8794` 26px: `MIT · one YAML · runs next to your teacher on one box`
  (traceable: `LICENSE`; README "fits next to a resident vLLM"). Settled 3.3s.
- **21.84s** — beneath that, smaller still, 22px, `#E0A458`, the callback:
  `The teacher is the ceiling.`
  Settled **2.2s** (floor 1.5s ✓). Holds alone as the music fades. No fade-to-black wipe — the
  frame simply ends with the caveat still legible.

Sequential/interaction: none. Long hold, big empty space.
Audio intent: one dry low impact on the title at 20.75s, then nothing. The footnote gets silence.
Audio-coupled idea: title impact only.
Music: fade 22.40s → 24.02s to zero.
Transition mood: end.

**Scene durations:** 3.82 + 3.82 + 5.47 + 4.36 + 2.72 + 3.83 = **24.02s** ✓ (15–25s)

**Music mood for this video:** deadpan — a low, steady bed that never celebrates anything.
**Audio summary:** near-silence under the failure line, the bed arrives with the product name,
one physical impact when the typed JSON lands on the 8.74s strong cue, two identical flat ticks
for the two numbers, one low impact on the title, and silence under the closing caveat.

---

## Two hook options (first 2 seconds) — user picks one

### Hook A — "The failure line" (CHOSEN, storyboarded above)
Black. One amber monospace line, no motion, no music accent:
> `On the hard Russian tasks the framework did not deliver.`
→ `— README.md · "Findings"`

*Why it wins:* it is verbatim, it is the single most unusual thing a launch video can say, it
cannot belong to any other project, and for the ML-engineer audience it buys credibility for
every number that follows. Risk: a viewer who watches only 2 seconds takes away a negative.

### Hook B — "The number with nothing behind it"
Black. One line of muted monospace at 0.27s: `labels used: 0` — hold 1.0s, dead still. At 1.3s,
directly beneath, in `#7CD4A0`, a second line lands: `accuracy vs real labels: 0.8825`. No third
line, no explanation until Scene 2.

*Why it might win:* it front-loads the strongest verified result and is the more shareable
thumbnail frame (`0` and `0.8825` read at any size). Risk: it is a *claim* first, which is the
posture the project's own README avoids — it recovers the tone only in Scene 6.

**If Hook B is chosen:** keep Scenes 2-6 exactly as storyboarded but move the failure line out of
Scene 1 and into Scene 6, above `The teacher is the ceiling.`, holding 2.0s; extend Scene 6 to
4.6s and trim Scene 4 row 2 (the 0.8825 row, now spent in the hook) to recover the time —
Scene 4 becomes 2.9s and the total lands at ~23.4s.
