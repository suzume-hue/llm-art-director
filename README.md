# LLM Art Director

**A language model can't hold a brush — but an art director never does. It looks, reasons, judges, and directs: studying an image, deciding what's wrong, and ordering edit after edit until it's better.**

LLM Art Director is an experiment in exactly that. It hands a large language model a set of *eyes* (a panel of aesthetic-scoring neural networks) and a set of *hands* (a darkroom of image-editing tools), then lets it direct the work: study an image, reason about what is holding it back, call for edits, check whether they actually helped, and keep going for a fixed number of rounds. A director with no ability to paint, refining an image by judgement alone.

This repository is the honest record of that idea — including the long stretch where it didn't work, and the one quiet lesson that turned out to matter more than any score.

---

## The premise

The starting thought was blunt:

> LLMs can't draw. Fine. They could, but maintaining that kind of pixel-level state would take forever. So they become artists by becoming *editors*. We have models that reason; we have models that judge composition and aesthetics. Let the reasoner read the judges, make decisions, apply edits, and adjust for *N* rounds — stopping early if it wants.

That's the whole philosophy. The art isn't in the brushwork. It's in the *seeing* and the *deciding*.

---

## What it actually does

You give it one image. It returns a refined version, plus a frame-by-frame record of how it got there.

Each round:

1. **Score** the current image with a panel of aesthetic models, then translate the raw numbers into plain language. The model reasons over sentences ("overall appeal: weak — your clearest opportunity"), never a table of floats.
2. **Reason and propose.** A vision LLM looks at the real image, names the weakness it actually *sees*, and proposes two or three competing edits.
3. **Verify by doing.** Every proposal, plus automatic strength variants, is applied and re-scored. The best one that clears the gate is kept. Ideas are tested, not trusted.
4. **Judge.** A human-preference model (the pretrained PickScore) compares the new state against the last approved one. If the numbers went up but a person would prefer the old version, the edit is **rolled back**.

That last step is the heart of the project, and it took ten versions to get right.

---

## The long road

This did not work for most of its life. The version history is the interesting part, so here it is honestly.

**v4–v5 — the skeleton.** Read an image, score it with LAION/NIMA/MUSIQ-style models, ask an LLM to pick an edit, apply it, repeat. The scaffolding existed but most of it was quietly broken.

**v6 — the bugs nobody saw.** The API-key rotation only caught one exception type, so the LLM calls never actually rotated keys; the "dynamic weighting" was dead code that never ran; a drift counter accumulated until it silently blocked every bold edit. Three features the comments proudly described and the program never executed.

**v7 — the artist was barely there.** A real run revealed the vision model failing with `Unterminated string` on almost every step and silently falling back to a *blind* text model. The cause: the new reasoning models spend tokens thinking, and the output budget was too small, so the JSON answer got cut off mid-sentence. The artist with eyes was hardly ever the one making decisions.

**v8 — letting it be bold.** Every dramatic edit was being killed by a structural-similarity gate that flagged a colour grade as "90% different" even though the subject never moved. Switched the gate to *semantic* drift (is it still the same scene?), so colour and mood became free to change while the picture stayed itself.

**v9 — the trap.** Added scorers actually trained on the image style. Composite shot up from 0.30 to 0.72. It looked like success. It wasn't: the model had discovered that cranking a vignette to maximum spiked the score, and the "best" result was a crushed, dark, ugly frame. The metric loved it. A human wouldn't.

**v10 — the judge was dead the whole time.** The anti-gaming guard meant to catch exactly the v9 failure had been returning `None` on every call since v7, silently doing nothing, because a model class quietly changed under it. A smoke test surfaced it. Once fixed, the human-preference judge finally fired on every step and rolled back the gaming, and the final source-vs-best check (a Gemini A/B) — for the first time — confirmed a result that beat its source. This is the version this repository ships.

**v11–v12 — the pivot.** A fair objection arrived: editing a fixed frame "feels nothing like art, it's just touching it up." So the project flipped. Instead of refining a given image, the model would *paint* one from a prompt, then branch into a tree of variations, deciding at each node whether to repaint or to retouch, and return a small gallery of finished works. That branch is its own line of work and lives elsewhere; this repo is the editor, kept whole.

---

## The lesson: metric vs. masterpiece

Here is a run on a quiet, sombre frame — a figure with their face buried in their arms. Fifteen accepted edits, the composite creeping up from 0.386 to 0.419 — unevenly, with plenty of dips on the way:

![the exploration of one frame](assets/run-recolour-evolution.png)

| The model's "best" | The scores, then the final check |
|---|---|
| ![recoloured result](assets/run-recolour-final.png) | composite **0.386 → 0.419** ✅ … final verdict: **source-preferred** ❌ |

The composite score rose, and the anime scorer rose with it (0.23 → 0.26), though the illustration scorer fell (0.88 → 0.52). Then the final check — a Gemini A/B comparison, run in both orderings — looked at the original and the result side by side and said, plainly, that the *original* was the better picture.

That disagreement is the whole project in one line. **Optimise any fixed measure of beauty hard enough and a model will find the cheap exploit that satisfies the measure while betraying the intent.** Oversharpening. Oversaturation. A heavy recolour into electric blue. A vignette cranked to black. Every guard in this codebase — the held-out validator, the semantic-drift gate, the per-metric weight cap, and above all the preference judge that can overrule the score — exists because an earlier version got caught doing precisely this.

The most useful result here is arguably a negative one: a small, reproducible demonstration of how readily metric-guided optimisation games its own metric, and what it takes to notice. (For an even blunter failure — what it does to a poster with text — see [Where it falls short](#where-it-falls-short-honestly) below.)

---

## How it works, in one diagram

```
                 ┌─────────────────────────────────────────────────────┐
  input image →  │  SCORE → READ (in words) → REASON → PROPOSE →        │
                 │     EXECUTE every candidate → JUDGE (human-pref)     │ → best image
                 └──────────▲──────────────────────────────────┬───────┘
                            └──────────── repeat N rounds ──────┘
```

- **Eyes** — Aesthetic Predictor v2.5, NIMA, MUSIQ, the LAION predictor, CLIP-IQA (held out), and PickScore (Kirstain et al.'s pretrained human-preference model, used as-is) as the judge. Optional illustration scorers for non-photographic input.
- **Hands** — parametric edits (exposure, contrast, curves, colour, clarity, grain, vignette, split-toning…), saliency-masked *local* edits (subject vs. background), and a structure-preserving generative editor (CosXL) for bold recolour/relight the dials can't reach. It edits the image; it never redraws it.
- **Brain** — a Gemini vision model as the artist, a Groq model as the blind critic, over plain REST with key rotation and an auto-ranked model roster that fails over on errors.
- **Conscience** — the four guards above.

---

## Run it

Built for a CUDA environment (developed on Kaggle's dual T4s: one GPU edits, one scores). Python 3.10+.

```bash
pip install -r requirements.txt
python art_director.py
```

Provide API keys for **Gemini** (artist) and **Groq** (critic), and optionally a **Hugging Face** token. On Kaggle, add them as Secrets named `GOOGLE_API_KEY_1..6`, `GROQ_API_KEY_1..5`, `HF_TOKEN`; locally, set the same names as environment variables. Point `INPUT_IMAGE_PATH` at your image and run. Outputs (best image, the `evolution/` frames, a contact sheet, and JSON logs) land in `/kaggle/working/`.

The script is laid out as a notebook (`# %%` cells) but runs as a plain file.

| Knob | Meaning |
|---|---|
| `INPUT_IMAGE_PATH` | the image to refine |
| `DOMAIN` | `"realistic"` (photo scorers) or `"anime"` (illustration scorers) |
| `MAX_EDIT_STEPS` | rounds to run |
| `USE_PICKSCORE_JUDGE` | the human-preference judge that rolls back gaming |
| `GENERATIVE_BACKEND` | `"cosxl"` (preferred) or `"ip2p"` |
| `SCORING_WEIGHTS` | the per-metric blend behind the composite |

---

## Where it falls short (honestly)

This section is the point. The project has real, structural problems, and pretending otherwise would waste the one thing it's good for — being a clear-eyed case study.

**It blurs and degrades the image.** Every generative edit runs the picture through a diffusion model at reduced resolution and scales it back, softening detail on each pass; even the parametric edits can mush texture. Outputs routinely come back *less* sharp than the source.

**It destroys text.** It has no concept of typography. Run it on the movie poster below and the title softens and the fine credit print smears into illegible mush — because the scorers reward tone and colour and are completely blind to whether words are still readable. It treats a poster as if it were a photograph with nothing written on it.

![what it did to a poster](assets/run-poster-final.png)

*A movie poster after refinement. The credits at the bottom are gone to soup; the title has lost its edges. The composite score was happy.*

**It often makes things worse.** Said plainly: more than once the "best" output by the system's own composite was judged worse than the original — by the final Gemini A/B check *or* by eye. A rising number frequently meant a degrading image. The sombre-portrait run earlier is exactly this: score up, picture worse.

**It games whatever you measure it by.** This is the defining flaw, not a footnote. Oversharpen, oversaturate, recolour everything electric blue, crush a vignette to black — the loop finds whatever cheap trick the metric rewards. The guards catch a lot. They do not catch all of it.

**The scorers are the wrong instrument for most things.** They're trained on photographs. On anime, illustration, posters, or anything stylised they sit in a narrow mid-band, barely move, and mislead. The composite is only meaningful on realistic photos, and even there it's a rough proxy. The "judge" that overrules it (PickScore) is itself a CLIP-based proxy — proxies refereeing proxies.

**It doesn't understand the image.** It reasons about tone, colour, and roughly where the subject is. It has no model of *content*, *intent*, or what the image is *for*. It can't decide "this is a poster, keep the text crisp" or "this is a portrait, protect the skin." Everything is a global knob-twiddling problem.

**It's slow, heavy, and fragile.** It wants two 16 GB GPUs and a stack of model downloads, and the reasoning leans on free-tier Gemini/Groq quotas that rate-limit and 503 mid-run. A bad API day produces a worse picture. Same image, different run, different result — it isn't reproducible.

**And the honest meta-point:** under the loop and the guards, it is mostly *prompting and plumbing*. It doesn't learn, it doesn't develop taste, and "artist" is a generous word. What it actually is: a search procedure that pokes at an image until some proxy numbers go up, with guards bolted on to stop the worst of its self-deception.

## So what is it good for

Not for finishing your posters. It's a working demonstration of a specific, important failure: that a reasoning loop will quietly game any fixed measure of beauty, and that catching it takes an independent judge and a stack of guards — and even then the gap between the metric and the masterpiece never fully closes. That negative result, reproducible and documented, is the real deliverable.

---

## License

[MIT](LICENSE) © Suzume

*An exploration of LLM-as-art-director: reasoning, decision-making, and the stubborn distance between a score and something worth looking at.*
