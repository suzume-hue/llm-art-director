# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🎨 Aesthetic Optimization Pipeline · v10
# Changes from v9 — a v9 run "worked" on paper (composite 0.30→0.72) but the
# vision judge said the ORIGINAL was better, and the run took forever. Three bugs
# + one design fix:
# - FIX (rate limits made it unusable): the auto-ranked premium Gemini models had
#   ZERO free-tier quota and 429'd every call (0 ok / 18 fail each), while one
#   model (flash-lite-latest) carried all 40 successes. v9 retried the dead models
#   FIRST every step and a 429 globally blocked the key for ALL models, so each
#   step burned minutes in 64s waits. v10: the roster LEARNS — a working model
#   floats to the front, failures get an escalating bench — and the caller tries
#   each model over the keys WITHOUT poisoning a key for other models, failing
#   fast to the next model instead of waiting.
# - FIX (the judge was silently dead): _pickscore() returned None on EVERY call in
#   v9 (a swallowed exception), so PickScore never gated anything — there is not
#   one ⚖ line in the v9 log. v10 surfaces the error and SMOKE-TESTS PickScore at
#   load; if it can't score the source it's disabled loudly (Gemini A/B fallback)
#   instead of pretending to work. (Also more robust feature extraction.)
# - FIX (one metric ran the show → vignette gaming): headroom weighting put 0.73
#   of the composite on the anime scorer alone, so the objective WAS that single
#   gameable number, and the system maximised it by cranking VIGNETTE to +1.0 —
#   ending on a crushed-dark vignetted frame the judge rejected. v10 caps any one
#   metric at MAX_METRIC_WEIGHT (0.40), caps destructive tools' strength
#   (vignette ≤0.55) and drops them from the ×1.45 up-sweep.
# - DESIGN (trust the artist over the metric): in v9 the artist's primary choice
#   was almost always thoughtful colour/atmosphere work, but the composite-max
#   selector overruled it with whatever scored highest (vignette). v10 prefers the
#   artist's PRIMARY candidate unless a sibling beats it by >PREFER_PRIMARY_MARGIN
#   — the artist's decision drives, the metric only breaks clear ties.
#
# Inherited from v9 (still in force):
# Changes from v8 — driven by a v8 run on an anime frame where the composite
# crept 0.504→0.559 and "wouldn't max out". The diagnosis: we were measuring the
# wrong thing.
# - WHY THE SCORES WON'T GO HIGH (the real answer): NIMA, MUSIQ, aesthetic-v2.5
#   and LAION are all trained on PHOTOGRAPHS. A stylised anime close-up is far
#   out of their distribution, so they park it near the middle of their range and
#   barely move — that's the "little deviation". And the ONE anime-trained judge
#   (shadowlilac) FAILED TO LOAD in the v8 run and fell back to a near-useless
#   binary classifier. We were optimising a ruler that doesn't fit the material.
# - NEW ANIME-AWARE SCORERS (now the primary signal): adds skytnt/anime-aesthetic
#   (a real anime aesthetic model, ONNX, self-contained preprocessing) and FIXES
#   the shadowlilac load (the pipeline couldn't auto-build an image processor —
#   v9 supplies one explicitly and tries mirrors). Photo scorers are DEMOTED to
#   minor guardrails; the anime scorers carry the weight.
# - HELD-OUT GUARD IS NOW ADVISORY, NOT A GATE: in the v8 run CLIP-IQA vetoed ~15
#   edits as "metric gaming" — but they were the artist correctly trying to warm
#   a garish-blue result. CLIP-IQA is photo-trained too, so it was a false
#   alarm. v9 still computes it and SHOWS it in the reasoning, but PickScore is
#   the only hard anti-gaming gate.
# - SCORES BECOME SENTENCES (you asked for this): the panel is now an art
#   director's READING — what each judge measures, that it's photo-trained so
#   mid-scale is expected on anime, what changed since the last edit, and the
#   clearest opportunity — and the artist writes its OWN visual verdict FIRST,
#   side by side with that reading. Decision-making over prose, not raw floats.
# - SEE THE EVOLUTION (you asked for this): every accepted edit's chosen image is
#   saved to /kaggle/working/evolution/ and assembled into one labelled contact
#   sheet (source → … → best) so you can see whether each stage contributed.
#
# Inherited from v8 (still in force):
# Changes from v7 — driven by an actual v7 run where "nothing changed":
# - FIX (the artist was barely running): in the v7 run, Gemini failed with
#   "Unterminated string" on almost EVERY step and silently fell back to the
#   BLIND Groq critic. The vision artist — the whole point — was mostly absent.
#   Cause: Gemini 3.x are thinking models; maxOutputTokens=2048 was eaten by
#   reasoning + 3 long rationales, truncating the JSON mid-string. v8: token
#   budget raised to 8192, rationales cut to one short clause, schema slimmed,
#   and a lenient JSON repair (`_loads_lenient`) closes truncated strings/braces
#   so a near-complete reply is still usable.
# - FIX (every bold edit was killed by drift → "nothing changed"): v7 rejected
#   color_grade/style_transfer at SSIM drift ~0.9 even when they IMPROVED the
#   composite (+0.024 at step 4!). SSIM measures luminance structure, so a colour
#   grade that keeps the same character/pose still reads as "0.9 drift". v8 gates
#   generative edits on CLIP SEMANTIC drift (is it still the same scene/subject?)
#   instead — colour, mood and tone are now FREE to change; only turning the
#   image into a different picture is blocked. SSIM kept as a loose backstop only.
# - CREATIVITY: generative tools are no longer a "last resort". The artist is
#   told it is free to reimagine palette, light and mood; bold strengths are
#   encouraged; the acceptance floor for generative edits is looser than for
#   parametric so experiments can land and be built on.
# - REASONING IN WORDS, NOT RAW NUMBERS: the score panel is now a natural-language
#   reading ("overall aesthetic appeal: weak — your best opportunity; richer,
#   moodier colour usually moves this") with only a tiny numeric footnote. Both
#   the artist and critic reason over prose.
# - MODEL FALLBACK + TRACKING: ranked rosters of Gemini and Groq models (built
#   from the live model list). On a model error the roster demotes it for a
#   cooldown and the call retries on the NEXT-best model — layered over the
#   existing per-model key rotation. Usage/failure counts per model are tracked
#   and printed at the end, and the model used is recorded on every decision.
# - (v8 had an optional "nano_banana" full-redraw backend — REMOVED in v9: an
#   artist edits with tools, it doesn't regenerate the picture from scratch.)
#
# Inherited from v7 (still in force):
# Changes from v6 — the four things that actually limit the system's ceiling:
# - EDITOR UPGRADE (GENERATIVE_BACKEND="cosxl"): InstructPix2Pix (SD1.5, 2023)
#   was the weakest link — its instruction-following is poor, so the artist's
#   intent rarely survived the edit. v7 loads CosXL Edit (SDXL-based instruction
#   editing) which fits a 16 GB T4 in fp16 and follows editing instructions far
#   better. Loaded defensively via from_single_file(is_cosxl_edit=True); if the
#   model is gated/unavailable or diffusers is too old, it FALLS BACK to IP2P so
#   the notebook still runs. Edits run at EDIT_RESOLUTION (768 on T4) then resize
#   back to the source resolution.
# - LOCAL / SPATIAL TOOLS (the big expressiveness gap): every v6 tool was global —
#   you could not "darken just the sky" or "sharpen just the subject", so fixing
#   one region wrecked another and the artist chased its tail. v7 adds a cheap
#   saliency map (cv2.saliency spectral-residual, no extra weights) and three
#   masked tools — subject_clarity, background_recede, subject_glow — that act
#   only where the subject is (or isn't). Auto-disabled if opencv-contrib is
#   missing.
# - A REAL JUDGE (PickScore, not Gemini-judging-Gemini): v6's reality-check had
#   the artist's own model family score its own work. v7 loads PickScore
#   (CLIP-ViT-H fine-tuned on ~1M human preference pairs) on the scoring GPU and
#   uses it as the rollback judge after every accepted edit — local, free, fast,
#   and trained specifically on "which image does a human prefer". The final
#   source-vs-best verdict still uses Gemini (a genuinely independent family).
#   Falls back to v6's periodic Gemini A/B if PickScore can't load.
# - GROUNDED PROMPTING (no more stale scene text): v6 described the SOURCE once
#   and refreshed every 5 steps, so the artist reasoned about edit N from a
#   snapshot taken 4 edits ago. v7 keeps the source description only as a fixed
#   anchor and tells the artist to LOOK at the current attached image each turn
#   and name the single biggest visual weakness it SEES before proposing. The
#   per-step refresh call is gone (also saves an API call every 5 steps).
#
# Everything below this line is inherited from v6 (still in force):
# Changes from v5 — bug fixes first:
# - FIX (key rotation was broken): v5's call_with_rotation only caught
#   requests.HTTPError, but Gemini went through the google-generativeai SDK and
#   Groq through the groq SDK — their rate-limit exceptions are different classes,
#   so 429s blew straight through and keys NEVER rotated. v6 calls BOTH providers
#   over plain REST (requests), so every 429/403 is caught and rotated correctly.
#   Bonus: the deprecated google-generativeai dependency is gone entirely, and we
#   get native JSON mode (responseMimeType / response_format) on both providers.
# - FIX (dynamic weights were dead code): _compute_run_weights() existed in v5 but
#   was never called — _RUN_WEIGHTS stayed empty forever. v6 wires it up after
#   source scoring AND recomputes the source composite under the run weights so
#   every Δcomposite in the run is an apples-to-apples comparison.
# - FIX (drift accumulation): _compute_scores grew a drift_ref param in v5
#   precisely to measure per-edit drift, but run_pipeline never passed it, so
#   accepted generative edits accumulated drift-from-source until the hard gate
#   blocked all future generative work. v6 gates per-edit drift (vs the image
#   being edited) AND a separate, looser cumulative cap (vs source).
# - FIX (perf): apply_hue_shift was a per-pixel Python loop over colorsys
#   (~8M calls on a 2MP image). Now vectorised through cv2 HSV.
# - IMPLEMENTED the v5 config that had no code behind it: FAST_RESTART_THRESHOLD
#   (immediate snap-back to best after a big accepted drop) and
#   BLOCKED_CATEGORY_* (categories empirically net-negative for a metric are
#   hard-blocked: filtered from candidates AND announced in the prompt).
#   USE_SDXL_GENERATIVE (flag with no backend) was removed.
#
# Architecture changes:
# - CRITIC DEMOTED (CRITIC_MODE="generative-only" by default): for PARAMETRIC
#   steps the blind text-only critic is skipped — empirical execution of every
#   candidate already supersedes its advice, and the two saved LLM calls per step
#   go further than a critique nobody can ground. The critic still reviews
#   GENERATIVE proposals (single-shot, expensive, highest-risk) and you can set
#   CRITIC_MODE="always" to restore v5 behaviour.
# - STRENGTH SWEEP: every LLM-proposed parametric candidate is auto-expanded with
#   ×0.65 and ×1.45 strength variants (deduped, capped at MAX_PARAMETRIC_EVALS).
#   Scoring is cheap; LLM calls are not — so each call now buys a small line
#   search, not one guess.
# - VISION REALITY-CHECK (anti reward-hacking): every VISION_CHECK_EVERY accepted
#   edits, Gemini A/B-compares the current image against the last visually
#   approved checkpoint (both orderings, to cancel position bias). If the
#   checkpoint wins both, the run ROLLS BACK and the artist is told its recent
#   streak was metric gaming (oversharpen/oversaturate is the classic failure of
#   NIMA/MUSIQ-style scorers). A final source-vs-best A/B verdict ships in the
#   result. This is the main defence against "scores went up, image got worse".
# - HELD-OUT VALIDATOR: CLIP-IQA (pyiqa) is scored on every candidate but kept
#   OUT of the composite and OUT of the prompts. An accepted edit may not drop
#   the held-out score by more than HELDOUT_TOLERANCE — a second, independent
#   guard against overfitting the four optimised metrics.
# - DIRECTION-AWARE TOOL PROFILE: effects are now keyed per direction
#   (contrast[+] vs contrast[-]) — v5 averaged opposite-sign effects into mush.
#   The profile also tracks Δcomposite per tool, which powers the blocking.
# - Early stop when every weighted metric reaches its target.
# - Loop-top rescoring removed (v5 re-ran all 5 scorers on the unchanged current
#   image every step); outputs saved as PNG (the v5 JPEG re-encode quietly threw
#   away part of the gain it had just optimised for).

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Using subprocess instead of !pip so this works in both .py and .ipynb contexts.
import subprocess, sys

_PKGS = [
    "diffusers", "transformers", "accelerate",
    "pandas", "tabulate", "xformers",
    "Pillow",
    # contrib build ships cv2.saliency (used by the new local/masked tools).
    # Installed instead of opencv-python-headless to avoid a double-install clash.
    "opencv-contrib-python-headless",
    "aesthetic-predictor-v2-5",   # → aesthetic_predictor_v2_5
    "pyiqa",                       # → NIMA, MUSIQ, CLIP-IQA
    "onnxruntime",                 # → skytnt/anime-aesthetic (domain scorer)
    "einops", "huggingface_hub",
]
print("Installing packages …")
_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *_PKGS],
    capture_output=True, text=True
)
if _result.returncode != 0:
    print(f"  [warn] pip stderr: {_result.stderr[-400:]}")
else:
    print("  ✓ All packages installed")

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
from pathlib import Path

INPUT_IMAGE_PATH: str = "/kaggle/input/datasets/supremedavid/sample-images/image_1.png"

GOOGLE_SECRET_NAMES: list[str] = [
    "GOOGLE_API_KEY_1", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3",
    "GOOGLE_API_KEY_4", "GOOGLE_API_KEY_5", "GOOGLE_API_KEY_6",
]
GROQ_SECRET_NAMES: list[str] = [
    "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3",
    "GROQ_API_KEY_4", "GROQ_API_KEY_5",
]
HF_SECRET_NAME: str = "HF_TOKEN"

# --- Model rosters (v9: auto-built from the FULL live list, not hand-picked) ---
# v9 no longer hardcodes a chosen subset. It takes EVERY model the API returns,
# drops the ones that can't do the job (image/audio/tts/embedding/etc.), and
# RANKS the rest automatically by a transparent heuristic (newest generation →
# pro>flash>lite → bigger context; for Groq, bigger parameter count). The whole
# roster is then used with demote-on-error fallback. You can still nudge with the
# optional lists/pins below, but by default the code arranges your list for you.
#
# Substring exclusion filters — a model whose id contains any of these is not a
# usable text/vision reasoner and is dropped from the artist/critic rosters.
GEMINI_EXCLUDE: list[str] = [
    "image", "tts", "audio", "embedding", "robotics", "computer-use",
    "live", "veo", "imagen", "lyria", "gemma", "aqa", "deep-research", "antigravity",
]
GROQ_EXCLUDE: list[str] = [
    "whisper", "orpheus", "tts", "guard", "allam", "compound",
]
# Optional manual front-loading (put a model first regardless of the heuristic).
# Empty = pure auto-ranking of your full list.
GEMINI_MODEL_PREFERENCE: list[str] = []
GROQ_MODEL_PREFERENCE:   list[str] = []
# Optional hard pins (use ONLY this model, skip the roster). None = use roster.
SELECTED_GEMINI_MODEL: str | None = None
SELECTED_GROQ_MODEL:   str | None = None
# How long to bench a model after it errors before retrying it.
MODEL_COOLDOWN_SECONDS: int = 90

# Gemini 3.x are thinking models — output tokens must cover reasoning AND the
# JSON answer or the reply truncates mid-string (the v7 "Unterminated string"
# bug). Give the artist room.
GEMINI_MAX_OUTPUT_TOKENS: int = 8192

DEVICE_EDIT:  str         = "cuda:0"
DEVICE_SCORE: str         = "cuda:1"
DTYPE:        torch.dtype = torch.float16

MAX_EDIT_STEPS:   int = 20
RESTART_PATIENCE: int = 6
MAX_RESTARTS:     int = 3
# v7: the per-step scene refresh is gone — the artist now reads the CURRENT
# attached image every turn instead of a stale cached description.

# Acceptance floor follows a cooling schedule: generous early (exploration),
# strict late (exploitation). See _exploration_floor().
EXPLORATION_FLOOR_START: float = -0.05
EXPLORATION_FLOOR_END:   float = -0.01
# v8 creativity: a GENERATIVE edit is a real artistic experiment, so it gets a
# much more generous floor than a parametric tweak — a bold recolour that the
# composite dislikes slightly is still worth landing and building on. The judge
# (PickScore) and semantic-drift gate are what keep it honest, not the floor.
GEN_EXPLORATION_FLOOR: float = -0.06
# Bias the artist toward bold, transformative moves rather than micro-edits.
CREATIVE_MODE: bool = True

# --- Drift gating (v8: semantic, not structural) ------------------------------
# THE v7 LESSON: SSIM (luminance structure) flagged colour grades at ~0.9 drift
# even though the subject/composition were untouched, so every bold edit died and
# "nothing changed". v8 gates GENERATIVE edits on CLIP SEMANTIC drift — "is this
# still the same scene and subject?" — which is invariant to colour/tone/mood.
# So the artist is free to recolour and relight; only morphing the picture into a
# genuinely different image is blocked. PARAMETRIC edits skip all drift gates.
SEM_DRIFT_HARD: float = 0.32   # per-edit hard ceiling — subject must survive
SEM_DRIFT_SOFT: float = 0.20   # per-edit soft zone — needs SEM_GAIN_MIN to pass
SEM_GAIN_MIN:   float = 0.010  # composite gain required inside the soft zone
CUM_SEM_DRIFT_CAP: float = 0.45  # total semantic departure from the source
# SSIM kept only as a loose backstop against total destruction (NOT the gate).
STRUCT_DRIFT_BACKSTOP: float = 0.92

MAX_IMAGE_PIXELS: int = 2048 * 2048
MIN_IMAGE_DIM:    int = 64
RPM_COOLDOWN: int = 65
RPD_COOLDOWN: int = 3600

# --- Scoring weights & normalization (v9: anime scorers carry the weight) -----
# THE v8 LESSON: photo-trained scorers (nima/musiq/aesthetic_v25/laion) sit
# mid-scale on anime and barely move. v9 adds anime-aware judges and makes THEM
# primary; the photo scorers are demoted to minor guardrails. anime_aesthetic
# (skytnt) + shadow_aesthetic (shadowlilac, now loading) together get ~55% of the
# weight. Headroom-proportional reweighting still applies on top.
SCORING_WEIGHTS: dict[str, float] = {
    "anime_aesthetic":  0.32,   # skytnt anime model — domain-appropriate, primary
    "shadow_aesthetic": 0.23,   # shadowlilac anime model (fixed loader)
    "aesthetic_v25":    0.16,
    "musiq":            0.13,
    "nima":             0.10,
    "laion_v2":         0.06,
}
NIMA_NORM_LO: float = 4.0
NIMA_NORM_HI: float = 7.0
AES_V25_NORM_LO: float = 2.0
AES_V25_NORM_HI: float = 8.5

# If shadowlilac fell back to cafeai/cafe_aesthetic (binary classifier, hard
# ceiling ~0.95, no gradient beyond ~0.85), suppress its weight.
SHADOW_BINARY_WEIGHT: float = 0.06

USE_DYNAMIC_WEIGHTS: bool = True
# v10: no single metric may exceed this share of the composite (was effectively
# 0.73 on the anime scorer in v9 → vignette gamed it). Keeps the objective a blend.
MAX_METRIC_WEIGHT: float = 0.40

# What constitutes "good" per metric (normalised 0→1). Used for headroom
# ranking, dynamic weights, and the early-stop check.
_SCORE_TARGETS: dict[str, float] = {
    "anime_aesthetic":  0.78,
    "shadow_aesthetic": 0.88,
    "aesthetic_v25":    0.70,
    "nima":             0.65,
    "musiq":            0.72,
    "laion_v2":         0.70,
}
EARLY_STOP_ON_TARGETS: bool = True

# --- Fast-restart trigger (implemented in v6) ---------------------------------
# If an accepted edit lands more than this below the current best composite,
# snap back to best immediately instead of waiting out RESTART_PATIENCE.
FAST_RESTART_THRESHOLD: float = 0.04

# --- Category blacklisting from tool-effect profile (implemented in v6) -------
# A (category[direction], metric) pair with ≥ MIN_N attempts and average Δ below
# THRESHOLD is hard-blocked: filtered from candidates and announced in prompts.
BLOCKED_CATEGORY_MIN_N: int  = 3
BLOCKED_CATEGORY_THRESHOLD: float = -0.012

LAION_V2_LOCAL: str = "/kaggle/working/ava+logos-l14-linearMSE.pth"
USE_PYIQA_NIMA:  bool = True
USE_PYIQA_MUSIQ: bool = True

# --- Held-out validation (anti reward-hacking guard #1) -----------------------
# CLIP-IQA is scored on every candidate but excluded from the composite and
# from every prompt. An accepted edit may not drop it by more than TOLERANCE.
# v8: generative edits get a looser tolerance — a bold recolour legitimately
# shifts technical-quality scores, and we don't want this guard to silently
# re-create the "nothing changes" problem the drift fix just solved.
# v9: CLIP-IQA is ADVISORY, not a hard gate. In the v8 run it vetoed ~15 edits
# as "metric gaming" that were actually the artist correctly warming a garish
# blue — and CLIP-IQA is photo-trained too, so the veto was a false alarm. It's
# still computed and SHOWN in the reasoning text, but PickScore is the only hard
# anti-gaming gate now. Set HELDOUT_AS_GATE=True to restore v8's hard veto.
USE_HELDOUT_VALIDATION: bool = True
HELDOUT_AS_GATE: bool = False
HELDOUT_TOLERANCE: float = 0.02
HELDOUT_TOLERANCE_GEN: float = 0.06

# --- Anime-aware aesthetic scorers (v9) ---------------------------------------
# skytnt/anime-aesthetic: a small ONNX model trained on anime images, single
# 0–1 aesthetic score. Self-contained preprocessing (we control it), so no HF
# image-processor headaches. Smoke-tested at load; auto-disabled if it misbehaves.
USE_ANIME_AESTHETIC: bool = True
ANIME_AESTHETIC_REPO: str = "skytnt/anime-aesthetic"
ANIME_AESTHETIC_FILE: str = "model.onnx"
# shadowlilac mirrors to try in order (the canonical v2 repo was removed). Each
# entry is (repo_id, hq_label). v9 loads these with an explicit image processor
# so the "Unrecognized image processor" failure from the v8 run is fixed.
SHADOW_MODEL_CANDIDATES: list[tuple] = [
    ("shadowlilac/aesthetic-shadow-v2", "hq"),
    ("NeoChen1024/aesthetic-shadow-v2-backup", "hq"),
    ("shadowlilac/aesthetic-shadow", "hq"),
]

# --- Evolution capture (v9): see every accepted stage --------------------------
SAVE_EVOLUTION: bool = True
EVOLUTION_DIR: str = "/kaggle/working/evolution"
EVOLUTION_THUMB: int = 360   # contact-sheet thumbnail size (px)
EVOLUTION_COLS: int = 5      # contact-sheet columns

# --- The judge (anti reward-hacking guard #2) ---------------------------------
# PickScore (CLIP-ViT-H fine-tuned on ~1M human preference pairs) is loaded
# locally and used as the rollback judge after EVERY accepted edit: if the new
# state is clearly less preferred than the last judge-approved checkpoint, the
# edit is rolled back and the artist is warned it was gaming the metrics. This
# replaces v6's "Gemini judges Gemini" check — PickScore is a different model
# family trained specifically on human preference, and being local it can gate
# every step for free instead of every 3rd. Falls back to the v6 periodic
# Gemini A/B if PickScore can't load. The FINAL source-vs-best verdict always
# uses Gemini (independent family, sanity cross-check).
USE_PICKSCORE_JUDGE: bool = True
PICKSCORE_REPO: str = "yuvalkirstain/PickScore_v1"
PICKSCORE_CLIP_REPO: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# Fixed aesthetic text condition — relative preference between two versions of
# the same photo is what matters, so a generic high-quality prompt is enough.
PICK_PROMPT: str = ("a high-quality, aesthetically pleasing, professionally photographed "
                    "and colour-graded image with natural tones and clean detail")
# Roll back only when the checkpoint is preferred with p > 0.5 + PICK_MARGIN
# (p = sigmoid of the PickScore difference). v8: widened so bold, tasteful edits
# survive — only a clear preference reversal (garish gaming) is undone.
PICK_MARGIN: float = 0.08
# Fallback only: if PickScore is unavailable, do a Gemini A/B every N accepted
# edits (v6 behaviour). Ignored when PickScore loads.
VISION_CHECK_EVERY: int = 3

# --- Critic policy -------------------------------------------------------------
# "generative-only" (default): blind critic only reviews GENERATIVE proposals —
#   parametric candidates are all executed anyway, which is a better critic.
# "always": v5 behaviour. "off": never.
CRITIC_MODE: str = "generative-only"

# --- "Propose multiple, verify by execution" (Idea2Img-style) ------------------
MAX_PARAMETRIC_CANDIDATES: int = 3        # LLM-proposed candidates per step
STRENGTH_SWEEP: tuple = (0.65, 1.0, 1.45) # auto strength variants per candidate
MAX_PARAMETRIC_EVALS: int = 6             # total apply+score budget per step
# v10 anti-gaming: these tools are destructive at high strength (heavy vignette
# crushed the v9 result to a dark vignetted frame the vision judge rejected). Cap
# their |strength| and don't let the ×1.45 up-sweep push them past the cap.
MAX_STRENGTH_BY_CATEGORY: dict[str, float] = {
    "vignette": 0.55, "contrast": 0.70, "curves_crush_blacks": 0.60, "clarity": 0.80,
}
NO_UPSWEEP_CATEGORIES: set = {"vignette", "contrast", "curves_crush_blacks"}
# v10: when several candidates are acceptable, prefer the ARTIST'S primary choice
# unless a sibling beats it by more than this composite margin — so the artist's
# intent drives, not pure metric-maximisation (which selected vignette in v9).
PREFER_PRIMARY_MARGIN: float = 0.03

# --- LLM sampling --------------------------------------------------------------
TEMP_ARTIST:   float = 0.85
TEMP_CRITIC:   float = 0.30
TEMP_DESCRIBE: float = 0.40
TEMP_JUDGE:    float = 0.10

# --- Tool-effect profile (compounding "understanding of the editor") -----------
TOOL_PROFILE_PATH: str = "/kaggle/working/tool_profile.json"
PRIOR_TOOL_PROFILE_PATH: str | None = None
TOOL_PROFILE_MIN_N: int = 1

# --- Generative editing backend -----------------------------------------------
# The artist edits with TOOLS — it does not get to regenerate the picture from
# scratch. v9 deliberately drops the "nano_banana" full-redraw backend (that's a
# different image, not an edit). Both options below are structure-preserving,
# instruction-conditioned darkroom tools:
# "cosxl" : CosXL Edit (SDXL instruction editing) — local, fits 16 GB T4.
# "ip2p"  : InstructPix2Pix (SD1.5) — original backend, weakest, lightest.
# Loader tries the requested backend, then falls back cosxl→ip2p.
GENERATIVE_BACKEND: str = "cosxl"
EDIT_RESOLUTION: int = 768   # long edge for the diffusion edit; 768 is T4-safe (1024 if you have headroom)
COSXL_REPO: str  = "stabilityai/cosxl"
COSXL_FILE: str  = "cosxl_edit.safetensors"

# Guidance scales, derived from |strength|. (image_guidance_scale, guidance_scale)
# anchors apply to BOTH backends — CosXL Edit and IP2P share the InstructPix2Pix
# sampling interface (higher image_guidance = stay closer to source; higher
# guidance = follow the text instruction harder).
IP2P_IMAGE_GUIDANCE_RANGE: tuple[float, float] = (1.70, 1.20)
IP2P_TEXT_GUIDANCE_RANGE:  tuple[float, float] = (5.0, 13.0)
IP2P_NUM_INFERENCE_STEPS:  int = 20
COSXL_IMAGE_GUIDANCE_RANGE: tuple[float, float] = (1.60, 1.10)
COSXL_TEXT_GUIDANCE_RANGE:  tuple[float, float] = (5.0, 9.0)
COSXL_NUM_INFERENCE_STEPS:  int = 20

# --- Local / spatial parametric tools -----------------------------------------
# Saliency-masked edits that act only on the subject (or only the background).
# Needs cv2.saliency (opencv-contrib). Auto-disabled at load if unavailable.
USE_LOCAL_TOOLS: bool = True

EDIT_CATEGORIES: dict[str, str] = {
    "brightness":           "PARAMETRIC",
    "contrast":             "PARAMETRIC",
    "tonal_contrast":       "PARAMETRIC",
    "clarity":              "PARAMETRIC",
    "curves_lift_shadows":  "PARAMETRIC",
    "curves_crush_blacks":  "PARAMETRIC",
    "color_temperature":    "PARAMETRIC",
    "saturation":           "PARAMETRIC",
    "hue_shift":            "PARAMETRIC",
    "sharpness":            "PARAMETRIC",
    "noise":                "PARAMETRIC",
    "film_grain_overlay":   "PARAMETRIC",
    "vignette":             "PARAMETRIC",
    "graduated_filter":     "PARAMETRIC",
    "vibrance":             "PARAMETRIC",
    "split_toning":         "PARAMETRIC",
    # local / spatial (saliency-masked) — may be removed at load if unsupported
    "subject_clarity":      "PARAMETRIC",
    "background_recede":    "PARAMETRIC",
    "subject_glow":         "PARAMETRIC",
    "style_transfer":       "GENERATIVE",
    "color_grade":          "GENERATIVE",
}

EDIT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "brightness":          "Overall exposure. Brighten (+) or darken (-) the scene.",
    "contrast":            "Global tonal range via CLAHE (+) or PIL flatten (-). Use before clarity.",
    "tonal_contrast":      "S-curve on luminance only — punches contrast without colour cast.",
    "clarity":             "Local midtone contrast — adds texture, presence, depth.",
    "curves_lift_shadows": "Open shadows, raise black point — airy, lifted feel.",
    "curves_crush_blacks": "Deepen blacks, cinematic shadow density.",
    "color_temperature":   "Warm (+) or cool (-) the atmosphere. Mood-defining.",
    "saturation":          "Enrich (+) or mute (-) all colours. Emotional intensity.",
    "hue_shift":           "Rotate the colour wheel. Subtle ±0.2, dramatic ±0.8.",
    "sharpness":           "Sharpen edges (+) or soften (-). Perceived detail.",
    "noise":               "Add grain (+) for analogue texture, or reduce noise (-).",
    "film_grain_overlay":  "Luminance-aware analogue grain — peaks in midtones.",
    "vignette":            "Darken (+) or lighten (-) frame edges. Compositional anchoring.",
    "graduated_filter":    "Darken top (+) or bottom (-) of frame. Sky/ground correction.",
    "vibrance":            "Selectively saturate muted colours; already-saturated hues are protected. Gentler than saturation.",
    "split_toning":        "Add a warm hue to highlights and cool hue to shadows (+), or invert (-). Cinematic colour separation.",
    "subject_clarity":     "LOCAL: sharpen/add clarity to the SUBJECT only (+), or soften it (-). Background untouched. Makes the subject pop without HDR-ing the whole frame.",
    "background_recede":   "LOCAL: push the BACKGROUND back — blur + slightly darken non-subject areas (+), or bring it forward (-). Subject stays sharp. Adds depth separation.",
    "subject_glow":        "LOCAL: brighten/lift the SUBJECT only (+, a dodge) or darken it (-). Background untouched. Directs the eye.",
    "style_transfer":      "[GENERATIVE] Deep stylistic transformation. Use when parametric plateaus.",
    "color_grade":         "[GENERATIVE] Full cinematic colour grade. Use when parametric plateaus.",
}

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, time, json, warnings, re, base64, io
from dataclasses import dataclass
from typing import Optional, Callable
from collections import Counter

import pandas as pd
import requests
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
import cv2

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"Devices  : {torch.cuda.device_count()}x GPU")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {props.name}  {props.total_memory // 1024**2} MB")

# Optional: seed the tool-effect profile from a previous run.
PRIOR_TOOL_PROFILE: dict = {}
if PRIOR_TOOL_PROFILE_PATH:
    try:
        with open(PRIOR_TOOL_PROFILE_PATH) as f:
            PRIOR_TOOL_PROFILE = json.load(f)
        _n_series = sum(len(m) for m in PRIOR_TOOL_PROFILE.values())
        print(f"✓ Loaded prior tool profile: {len(PRIOR_TOOL_PROFILE)} categories, {_n_series} metric-series")
    except Exception as e:
        print(f"  [skip] prior tool profile — {e}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
def validate_input_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():   raise ValueError(f"Input image not found: {path}")
    if not p.is_file():  raise ValueError(f"Not a file: {path}")
    if p.suffix.lower() not in {".jpg",".jpeg",".png",".webp",".bmp",".tiff"}:
        raise ValueError(f"Unsupported format: {p.suffix}")
    try:
        img = Image.open(p).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image: {e}") from e
    w, h = img.size
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM: raise ValueError(f"Too small ({w}×{h})")
    if w * h > MAX_IMAGE_PIXELS:               raise ValueError(f"Too large ({w}×{h})")
    print(f"✓ Input: {p.name}  ({w}×{h})")
    return img

source_image: Image.Image = validate_input_image(INPUT_IMAGE_PATH)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def _load_secrets(names: list[str]) -> list[str]:
    keys: list[str] = []
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        _get = lambda n: client.get_secret(n)
    except ImportError:
        _get = lambda n: os.environ.get(n, "")
    for name in names:
        try:
            v = _get(name)
            if v and v.strip(): keys.append(v.strip())
            else: print(f"  [skip] '{name}' — empty")
        except Exception as e:
            print(f"  [skip] '{name}' — {e}")
    return keys

print("Loading Google keys …")
GOOGLE_KEYS: list[str] = _load_secrets(GOOGLE_SECRET_NAMES)
print(f"  → {len(GOOGLE_KEYS)} key(s)")
print("Loading Groq keys …")
GROQ_KEYS: list[str]   = _load_secrets(GROQ_SECRET_NAMES)
print(f"  → {len(GROQ_KEYS)} key(s)")
print("Loading HF token …")
_hf_list   = _load_secrets([HF_SECRET_NAME])
HF_TOKEN: str | None   = _hf_list[0] if _hf_list else None
print(f"  → {'loaded' if HF_TOKEN else 'NOT FOUND — LAION V2 may fail'}")

if not GOOGLE_KEYS: raise RuntimeError("No Google API keys found.")
if not GROQ_KEYS:   raise RuntimeError("No Groq API keys found.")

# %% [code] {"jupyter":{"outputs_hidden":false}}
@dataclass
class _KeyState:
    key: str
    rpm_blocked: bool  = False
    rpd_blocked: bool  = False
    rpm_unblock_at: float = 0.0
    rpd_unblock_at: float = 0.0
    @property
    def available(self) -> bool: return not self.rpm_blocked and not self.rpd_blocked

class KeyRotator:
    """All LLM traffic in v6 goes over plain REST (requests), so every provider
    error is a requests.HTTPError and rotation actually fires — in v5 the SDK
    exception classes slipped past this handler and the keys never rotated."""
    def __init__(self, keys: list[str], rpm_cd: int, rpd_cd: int):
        self._pool = [_KeyState(k) for k in keys]
        self._rpm_cd = rpm_cd; self._rpd_cd = rpd_cd; self._current = -1

    def get(self) -> str:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            self._refresh()
            k = self._next()
            if k: return k
            w = self._min_wait()
            print(f"  [rate-limit] all keys capped — waiting {w:.0f}s …")
            time.sleep(w + 1)
        raise RuntimeError("All keys rate-limited.")

    def call_with_rotation(self, fn: Callable, *, max_attempts: int | None = None):
        attempts = max_attempts or len(self._pool) + 1
        backoff = 1.0; last_err = None
        for _ in range(attempts):
            key = self.get()
            try:
                return fn(key)
            except requests.HTTPError as e:
                last_err = e
                code = e.response.status_code if e.response is not None else 0
                if code == 429:   self.mark_rpm(key)
                elif code == 403: self.mark_rpd(key)
                elif code >= 500:  # transient server error — retry, don't punish the key
                    print(f"  [server {code}] retry in {backoff:.0f}s")
                    time.sleep(backoff); backoff = min(backoff*2, 60.0)
                else: raise
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                print(f"  [net] {type(e).__name__} — retry in {backoff:.0f}s")
                time.sleep(backoff); backoff = min(backoff*2, 60.0)
        raise RuntimeError(f"All attempts exhausted. Last: {last_err}")

    def mark_rpm(self, key: str):
        s = self._find(key)
        if s: s.rpm_blocked=True; s.rpm_unblock_at=time.monotonic()+self._rpm_cd
    def mark_rpd(self, key: str):
        s = self._find(key)
        if s: s.rpd_blocked=True; s.rpd_unblock_at=time.monotonic()+self._rpd_cd
    def _refresh(self):
        now = time.monotonic()
        for s in self._pool:
            if s.rpm_blocked and now>=s.rpm_unblock_at: s.rpm_blocked=False
            if s.rpd_blocked and now>=s.rpd_unblock_at: s.rpd_blocked=False
    def _next(self) -> str | None:
        n = len(self._pool)
        for _ in range(n):
            self._current = (self._current+1)%n
            if self._pool[self._current].available: return self._pool[self._current].key
        return None
    def _find(self, key: str) -> Optional[_KeyState]:
        for s in self._pool:
            if s.key == key: return s
        return None
    def _min_wait(self) -> float:
        now = time.monotonic(); cands = []
        for s in self._pool:
            if s.rpm_blocked: cands.append(s.rpm_unblock_at-now)
            if s.rpd_blocked: cands.append(s.rpd_unblock_at-now)
        return max(0.0, min(cands)) if cands else float(self._rpm_cd)

google_rotator = KeyRotator(GOOGLE_KEYS, RPM_COOLDOWN, RPD_COOLDOWN)
groq_rotator   = KeyRotator(GROQ_KEYS,   RPM_COOLDOWN, RPD_COOLDOWN)
print(f"Rotators ready. Google:{len(GOOGLE_KEYS)}  Groq:{len(GROQ_KEYS)}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
def _likely_vision(name: str) -> bool:
    n = name.lower()
    return "gemini" in n and not any(s in n for s in ("gemma","embedding","tts","veo","imagen","lyria","robotics"))

def list_gemini_models(api_key: str) -> pd.DataFrame:
    url  = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=200"
    resp = requests.get(url, timeout=15); resp.raise_for_status()
    rows = []
    for m in resp.json().get("models", []):
        name    = m.get("name","")
        methods = m.get("supportedGenerationMethods",[])
        rows.append({
            "model_id":     name,
            "display_name": m.get("displayName",""),
            "input_tokens": m.get("inputTokenLimit","—"),
            "output_tokens":m.get("outputTokenLimit","—"),
            "vision":       "✓" if "generateContent" in methods and _likely_vision(name) else "·",
            "methods":      ", ".join(methods),
        })
    return pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)

gemini_df = google_rotator.call_with_rotation(list_gemini_models)
print(f"\n{'='*70}\n  GEMINI MODELS ({len(gemini_df)})   ✓ = vision-capable\n{'='*70}")
with pd.option_context("display.max_rows",100,"display.max_colwidth",60):
    print(gemini_df[["model_id","display_name","vision","input_tokens"]].to_string())
print("\n→ Set SELECTED_GEMINI_MODEL in CELL 2 to a ✓ vision model.")

# %% [code] {"jupyter":{"outputs_hidden":false}}
def list_groq_models(api_key: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp    = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=15)
    resp.raise_for_status()
    rows = [{"model_id":m.get("id",""),"owned_by":m.get("owned_by","—"),
             "context_window":m.get("context_window","—"),"active":"✓" if m.get("active",True) else "·"}
            for m in resp.json().get("data",[])]
    return pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)

groq_df = groq_rotator.call_with_rotation(list_groq_models)
print(f"\n{'='*70}\n  GROQ MODELS ({len(groq_df)})\n{'='*70}")
with pd.option_context("display.max_rows",60,"display.max_colwidth",50):
    print(groq_df.to_string())
print("\n→ Set SELECTED_GROQ_MODEL in CELL 2.")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# v9: build the rosters AUTOMATICALLY from the FULL live model list — keep every
# usable model and arrange them by a transparent heuristic, instead of a
# hand-picked subset.
import re as _re_rank

def _gemini_rank_key(mid: str, ctx) -> tuple:
    """Higher = better. Newest generation first, then pro>flash>lite, then
    context window. 'latest' aliases are treated as current-generation."""
    n = mid.lower()
    m = _re_rank.search(r"gemini-(\d+(?:\.\d+)?)", n)
    ver = float(m.group(1)) if m else 0.0
    if "latest" in n: ver = max(ver, 3.4)          # current-gen alias
    tier = 2 if "pro" in n else (0 if "lite" in n else 1)   # pro > flash > lite
    preview = -0.15 if "preview" in n else 0.0
    dup = -0.05 if _re_rank.search(r"-\d{3}$", n) else 0.0   # demote -001 dupes
    try: ctx = int(ctx)
    except Exception: ctx = 0
    return (ver, tier, preview, dup, ctx)

def _groq_rank_key(mid: str, ctx) -> tuple:
    """Higher = better. Bigger parameter count first, then context window."""
    n = mid.lower()
    m = _re_rank.search(r"(\d+)\s*b", n)
    size = int(m.group(1)) if m else 0
    fam = 1 if ("gpt-oss" in n or "llama-4" in n or "llama-3.3" in n or "qwen" in n) else 0
    try: ctx = int(ctx)
    except Exception: ctx = 0
    return (fam, size, ctx)

def _usable(mid: str, exclude: list[str]) -> bool:
    n = mid.lower()
    return not any(x in n for x in exclude)

def _build_roster(df, id_col, ctx_col, exclude, rank_key, vision_ok=None,
                  front: list[str] | None = None) -> list[str]:
    rows = [(r[id_col], r.get(ctx_col, 0)) for _, r in df.iterrows()]
    cands = [(mid, ctx) for mid, ctx in rows
             if _usable(mid, exclude) and (vision_ok is None or mid in vision_ok)]
    cands.sort(key=lambda t: rank_key(t[0], t[1]), reverse=True)
    ordered = [mid for mid, _ in cands]
    # optional manual front-loading (substring match), preserving the rest
    if front:
        head = [m for want in front for m in ordered if want in m]
        seen = set(); head = [m for m in head if not (m in seen or seen.add(m))]
        ordered = head + [m for m in ordered if m not in head]
    return ordered

_gem_ids      = gemini_df["model_id"].tolist()
_gem_vision   = set(gemini_df.loc[gemini_df["vision"]=="✓","model_id"].tolist())
_groq_ids     = groq_df["model_id"].tolist()

if SELECTED_GEMINI_MODEL:   # hard pin overrides the roster
    GEMINI_ROSTER=[SELECTED_GEMINI_MODEL]
else:
    GEMINI_ROSTER=_build_roster(gemini_df,"model_id","input_tokens",GEMINI_EXCLUDE,
                                _gemini_rank_key, vision_ok=_gem_vision,
                                front=GEMINI_MODEL_PREFERENCE)
if SELECTED_GROQ_MODEL:
    GROQ_ROSTER=[SELECTED_GROQ_MODEL]
else:
    GROQ_ROSTER=_build_roster(groq_df,"model_id","context_window",GROQ_EXCLUDE,
                              _groq_rank_key, vision_ok=None,
                              front=GROQ_MODEL_PREFERENCE)

if not GEMINI_ROSTER:
    raise ValueError(f"No usable vision Gemini model after excludes {GEMINI_EXCLUDE}.\n"
                     f"  Available vision models: {sorted(_gem_vision)}")
if not GROQ_ROSTER:
    raise ValueError(f"No usable Groq model after excludes {GROQ_EXCLUDE}.\n  Available: {_groq_ids}")

print(f"✓ Gemini roster ({len(GEMINI_ROSTER)}, auto-ranked from full list):")
for i,m in enumerate(GEMINI_ROSTER): print(f"    {i+1:2d}. {m}")
print(f"✓ Groq roster   ({len(GROQ_ROSTER)}, auto-ranked from full list):")
for i,m in enumerate(GROQ_ROSTER): print(f"    {i+1:2d}. {m}")

_prior_n = sum(len(m) for m in PRIOR_TOOL_PROFILE.values()) if PRIOR_TOOL_PROFILE else 0
print("┌──────────────────────────────────────────────────────────────┐")
print("│  Active configuration (v10)                                    │")
print("├──────────────────────────────────────────────────────────────┤")
print(f"│  Gemini roster     : {f'{GEMINI_ROSTER[0].split(chr(47))[-1]} +{len(GEMINI_ROSTER)-1} more':<42} │")
print(f"│  Groq roster       : {f'{GROQ_ROSTER[0]} +{len(GROQ_ROSTER)-1} more':<42} │")
print(f"│  Editor backend    : {f'{GENERATIVE_BACKEND} @ {EDIT_RESOLUTION}px (→cosxl→ip2p)':<42} │")
print(f"│  Judge             : {(f'PickScore (local) → Gemini final' if USE_PICKSCORE_JUDGE else f'Gemini A/B every {VISION_CHECK_EVERY}'):<42} │")
print(f"│  Anime scorers     : {f'skytnt={USE_ANIME_AESTHETIC}, shadowlilac on':<42} │")
print(f"│  Local tools       : {('saliency-masked (subject/bg)' if USE_LOCAL_TOOLS else 'OFF'):<42} │")
print(f"│  Critic mode       : {CRITIC_MODE:<42} │")
print(f"│  Drift gate        : {f'semantic {SEM_DRIFT_HARD}/{SEM_DRIFT_SOFT}, cum {CUM_SEM_DRIFT_CAP}':<42} │")
print(f"│  Creative mode     : {('ON — bold edits encouraged' if CREATIVE_MODE else 'off'):<42} │")
print(f"│  Max steps         : {MAX_EDIT_STEPS:<42} │")
print(f"│  Floor (P/G)       : {f'{EXPLORATION_FLOOR_START}→{EXPLORATION_FLOOR_END} / gen {GEN_EXPLORATION_FLOOR}':<42} │")
print(f"│  Evals/step        : {f'{MAX_PARAMETRIC_CANDIDATES} LLM cand × sweep, cap {MAX_PARAMETRIC_EVALS}':<42} │")
print(f"│  Held-out guard    : {('CLIP-IQA ADVISORY (not a gate)' if USE_HELDOUT_VALIDATION and not HELDOUT_AS_GATE else 'CLIP-IQA gate' if USE_HELDOUT_VALIDATION else 'OFF'):<42} │")
print(f"│  Evolution capture : {(EVOLUTION_DIR if SAVE_EVOLUTION else 'OFF'):<42} │")
print(f"│  Prior tool profile: {(f'{len(PRIOR_TOOL_PROFILE)} cats, {_prior_n} series' if PRIOR_TOOL_PROFILE else 'none (fresh)'):<42} │")
print("└──────────────────────────────────────────────────────────────┘")

# %% [code] {"jupyter":{"outputs_hidden":false}}
def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.float32) / 255.0

def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr.clip(0,1)*255).astype(np.uint8))

def _make_scurve_lut(strength: float) -> np.ndarray:
    x = np.linspace(0,1,256)
    k = 5.0*abs(strength); mid = 0.5
    y = 1.0/(1.0+np.exp(-k*(x-mid)))
    y = (y-y[0])/(y[-1]-y[0])
    if strength < 0: y = y*(1-abs(strength))+x*abs(strength)
    return (y*255).astype(np.uint8)

def apply_brightness(img, s):
    return ImageEnhance.Brightness(img).enhance(max(0.0,1.0+s))

def apply_contrast(img, s):
    if s > 0:
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0+s*3.0, tileGridSize=(8,8))
        return Image.fromarray(cv2.cvtColor(cv2.merge([clahe.apply(l),a,b]), cv2.COLOR_LAB2RGB))
    return ImageEnhance.Contrast(img).enhance(max(0.0,1.0+s))

def apply_tonal_contrast(img, s):
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    return Image.fromarray(cv2.cvtColor(cv2.merge([cv2.LUT(l,_make_scurve_lut(s)),a,b]), cv2.COLOR_LAB2RGB))

def apply_clarity(img, s):
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    if s > 0:
        blurred = cv2.GaussianBlur(l.astype(np.float32),(0,0),sigmaX=10)
        l = np.clip(l.astype(np.float32)+s*0.5*(l.astype(np.float32)-blurred),0,255).astype(np.uint8)
    else:
        l = cv2.GaussianBlur(l,(0,0),sigmaX=max(0.5,abs(s)*5))
    return Image.fromarray(cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB))

def apply_curves_lift_shadows(img, s):
    arr = _pil_to_np(img)
    if s >= 0: arr = arr*(1.0-s*0.15)+s*0.15
    else:      arr = np.power(arr.clip(1e-6,1.0),1.0+abs(s)*1.5)
    return _np_to_pil(arr.clip(0,1))

def apply_curves_crush_blacks(img, s):
    arr = _pil_to_np(img)
    if s > 0: arr = np.power(arr.clip(1e-6,1.0),1.0+s*1.5)
    else:     arr = arr*(1.0-abs(s)*0.10)+abs(s)*0.10
    return _np_to_pil(arr.clip(0,1))

def apply_color_temperature(img, s):
    arr = _pil_to_np(img)
    warm,cool = max(0.0,s),max(0.0,-s)
    arr[...,0] = (arr[...,0]*(1+0.30*warm)).clip(0,1)
    arr[...,1] = (arr[...,1]*(1+0.10*warm)).clip(0,1)
    arr[...,2] = (arr[...,2]*(1+0.30*cool)).clip(0,1)
    return _np_to_pil(arr)

def apply_saturation(img, s):
    return ImageEnhance.Color(img).enhance(max(0.0,1.0+s))

def apply_hue_shift(img, s):
    # v6: vectorised via cv2 (v5 looped colorsys over every pixel — ~8M Python
    # calls on a 2MP image). cv2 hue channel is 0–179 (degrees/2): s*30° = s*15.
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[...,0] = (hsv[...,0] + int(round(s*15))) % 180
    return Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))

def apply_sharpness(img, s):
    if s >= 0: return img.filter(ImageFilter.UnsharpMask(radius=2,percent=int(50+s*100)))
    return ImageEnhance.Sharpness(img).enhance(max(0.0,1.0+s))

def apply_noise(img, s):
    if s > 0:
        arr = _pil_to_np(img)
        return _np_to_pil((arr+np.random.normal(0,s*0.08,arr.shape).astype(np.float32)).clip(0,1))
    arr = np.array(img)
    return Image.fromarray(cv2.medianBlur(arr, max(3,int(abs(s)*10)|1)))

def apply_film_grain_overlay(img, s):
    if s <= 0: return apply_noise(img, s)
    arr = _pil_to_np(img)
    lum  = arr.mean(axis=2,keepdims=True)
    mask = 4.0*lum*(1.0-lum)
    grain = np.random.normal(0,s*0.06,arr.shape).astype(np.float32)*mask
    return _np_to_pil((arr+grain).clip(0,1))

def apply_vignette(img, s):
    w,h = img.size
    mask = Image.new("L",(w,h),0); draw = ImageDraw.Draw(mask)
    for i in range(60):
        t=i/60; pad=int(min(w,h)*0.5*t)
        draw.rectangle([pad,pad,w-pad,h-pad],fill=255-int(abs(s)*255*t))
    m = np.array(mask,dtype=np.float32)/255.0
    arr = _pil_to_np(img)
    if s > 0: arr = arr*m[...,np.newaxis]
    else:     arr = arr+(1-arr)*(1-m[...,np.newaxis])*abs(s)
    return _np_to_pil(arr)

def apply_graduated_filter(img, s):
    arr = _pil_to_np(img); h,w = arr.shape[:2]
    if s > 0: grad = np.linspace(1-abs(s)*0.6,1.0,h)[:,np.newaxis,np.newaxis]
    else:     grad = np.linspace(1.0,1-abs(s)*0.6,h)[:,np.newaxis,np.newaxis]
    return _np_to_pil((arr*np.broadcast_to(grad,arr.shape)).clip(0,1))

def apply_vibrance(img, s):
    """Selective saturation: boosts muted colours more than vivid ones."""
    arr = _pil_to_np(img)
    hsv = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:,:,1]
    protect = sat / 255.0          # 0=muted (full boost), 1=vivid (protected)
    boost = (1.0 - protect) * abs(s) * 80.0
    if s > 0: hsv[:,:,1] = np.clip(sat + boost, 0, 255)
    else:     hsv[:,:,1] = np.clip(sat - boost, 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb)

def apply_split_toning(img, s):
    """Warm highlights + cool shadows (s > 0) or the inverse (s < 0)."""
    arr = _pil_to_np(img)
    lum = arr.mean(axis=2, keepdims=True)
    strength = abs(s) * 0.18
    if s > 0:
        shadow_mask    = np.clip(1.0 - lum * 2.5, 0, 1)
        highlight_mask = np.clip(lum * 2.5 - 1.5, 0, 1)
    else:
        shadow_mask    = np.clip(lum * 2.5 - 1.5, 0, 1)
        highlight_mask = np.clip(1.0 - lum * 2.5, 0, 1)
    arr[...,0] = np.clip(arr[...,0] - shadow_mask[...,0] * strength, 0, 1)
    arr[...,1] = np.clip(arr[...,1] + shadow_mask[...,0] * strength * 0.3, 0, 1)
    arr[...,2] = np.clip(arr[...,2] + shadow_mask[...,0] * strength * 0.7, 0, 1)
    arr[...,0] = np.clip(arr[...,0] + highlight_mask[...,0] * strength * 0.8, 0, 1)
    arr[...,1] = np.clip(arr[...,1] + highlight_mask[...,0] * strength * 0.3, 0, 1)
    arr[...,2] = np.clip(arr[...,2] - highlight_mask[...,0] * strength * 0.5, 0, 1)
    return _np_to_pil(arr.clip(0, 1))

# --- Local / spatial tools (saliency-masked) ----------------------------------
# A subject mask from cv2's spectral-residual saliency (no extra weights). The
# raw map is coarse, so it's heavily feathered and used only as a soft blend
# weight — enough to bias an edit toward the subject or the background without
# needing a real segmentation model. v6 had only global tools, which is why the
# artist kept fixing one region and wrecking another.
_saliency_detector = None
if USE_LOCAL_TOOLS:
    try:
        _saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        # smoke-test on a tiny array so we fail HERE, not mid-run
        _ok, _ = _saliency_detector.computeSaliency(np.zeros((32,32,3), np.uint8))
        if not _ok: raise RuntimeError("computeSaliency returned False")
        print("  ✓ Saliency detector ready (local tools enabled)")
    except Exception as e:
        print(f"  [skip] local tools — cv2.saliency unavailable ({e}); install opencv-contrib")
        _saliency_detector = None

def _subject_mask(img: Image.Image) -> np.ndarray:
    """Feathered subject mask in [0,1], HxW float32 (high = subject)."""
    arr = np.array(img)
    ok, sal = _saliency_detector.computeSaliency(arr)
    if not ok: return np.ones(arr.shape[:2], np.float32)
    sal = sal.astype(np.float32)
    feather = max(3, int(max(arr.shape[:2]) * 0.02) | 1)
    sal = cv2.GaussianBlur(sal, (feather, feather), 0)
    lo, hi = float(sal.min()), float(sal.max())
    sal = (sal - lo) / (hi - lo + 1e-6)
    return sal

def _blend(base: np.ndarray, edited: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    m = mask2d[..., None]
    return (base * (1.0 - m) + edited * m).clip(0, 1)

def apply_subject_clarity(img, s):
    """Clarity/sharpen on the subject only (+) or soften it (-)."""
    base = _pil_to_np(img)
    edited = _pil_to_np(apply_clarity(img, s))
    mask = _subject_mask(img) * min(1.0, abs(s) + 0.15)
    return _np_to_pil(_blend(base, edited, mask))

def apply_background_recede(img, s):
    """Push the background back: blur + slight darken on non-subject (+),
    or bring it forward: mild local contrast on non-subject (-)."""
    base = _pil_to_np(img)
    if s >= 0:
        sigma = max(0.6, abs(s) * 6.0)
        blurred = cv2.GaussianBlur((base*255).astype(np.uint8), (0,0), sigmaX=sigma).astype(np.float32)/255.0
        darkened = blurred * (1.0 - abs(s) * 0.18)
        bg_mask = (1.0 - _subject_mask(img)) * min(1.0, abs(s) + 0.10)
        return _np_to_pil(_blend(base, darkened, bg_mask))
    edited = _pil_to_np(apply_clarity(img, abs(s)*0.6))
    bg_mask = (1.0 - _subject_mask(img)) * min(1.0, abs(s) + 0.10)
    return _np_to_pil(_blend(base, edited, bg_mask))

def apply_subject_glow(img, s):
    """Dodge (brighten) the subject (+) or burn (darken) it (-)."""
    base = _pil_to_np(img)
    edited = (base * (1.0 + s * 0.35)).clip(0, 1)
    mask = _subject_mask(img) * min(1.0, abs(s) + 0.15)
    return _np_to_pil(_blend(base, edited, mask))

PARAMETRIC_TRANSFORMS = {
    "brightness":          apply_brightness,
    "contrast":            apply_contrast,
    "tonal_contrast":      apply_tonal_contrast,
    "clarity":             apply_clarity,
    "curves_lift_shadows": apply_curves_lift_shadows,
    "curves_crush_blacks": apply_curves_crush_blacks,
    "color_temperature":   apply_color_temperature,
    "saturation":          apply_saturation,
    "hue_shift":           apply_hue_shift,
    "sharpness":           apply_sharpness,
    "noise":               apply_noise,
    "film_grain_overlay":  apply_film_grain_overlay,
    "vignette":            apply_vignette,
    "graduated_filter":    apply_graduated_filter,
    "vibrance":            apply_vibrance,
    "split_toning":        apply_split_toning,
}
if _saliency_detector is not None:
    PARAMETRIC_TRANSFORMS.update({
        "subject_clarity":   apply_subject_clarity,
        "background_recede": apply_background_recede,
        "subject_glow":      apply_subject_glow,
    })
else:
    # Local tools unsupported — drop them from the menu so the artist never
    # proposes a tool that doesn't exist.
    for _k in ("subject_clarity", "background_recede", "subject_glow"):
        EDIT_CATEGORIES.pop(_k, None); EDIT_TOOL_DESCRIPTIONS.pop(_k, None)

def apply_edit(img: Image.Image, category: str, strength: float) -> Image.Image:
    if EDIT_CATEGORIES.get(category) == "GENERATIVE":
        raise NotImplementedError(f"'{category}' is GENERATIVE — route to the generative backend.")
    if category not in PARAMETRIC_TRANSFORMS:
        raise ValueError(f"Unknown category: '{category}'")
    return PARAMETRIC_TRANSFORMS[category](img, strength)

print(f"Parametric transforms ready ({len(PARAMETRIC_TRANSFORMS)} tools, "
      f"local={'on' if _saliency_detector is not None else 'off'}).")

# %% [code] {"jupyter":{"outputs_hidden":false}}
from transformers import CLIPProcessor, CLIPModel, pipeline as hf_pipeline

# CLIP: LAION V2 features AND v8 SEMANTIC drift (is it still the same scene?).
print("Loading CLIP ViT-L/14 …")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", token=HF_TOKEN).to(DEVICE_SCORE).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", token=HF_TOKEN)
print("  ✓ CLIP  (LAION V2 scoring + semantic drift)")

@torch.no_grad()
def clip_embed(img: Image.Image) -> torch.Tensor:
    inputs = clip_processor(images=img, return_tensors="pt")
    emb    = clip_model.get_image_features(pixel_values=inputs["pixel_values"].to(DEVICE_SCORE))
    if not isinstance(emb, torch.Tensor):
        emb = emb.image_embeds if hasattr(emb,"image_embeds") else emb.pooler_output
    return F.normalize(emb, dim=-1)

def _clip_drift(img1: Image.Image, img2: Image.Image) -> float:
    """Semantic distance = 1 - cosine(CLIP(img1), CLIP(img2)). ~0 for a recolour
    of the same scene; large only when the picture's CONTENT changes. This is the
    v8 generative gate — it lets colour/tone/mood move freely while still
    catching an edit that turns the image into a different picture."""
    a=clip_embed(img1); b=clip_embed(img2)
    return float(max(0.0, 1.0 - float((a*b).sum().item())))

def _ssim_drift(img1: Image.Image, img2: Image.Image) -> float:
    """SSIM structural distance. 0=identical, 1=completely different."""
    if img1.size != img2.size: img2 = img2.resize(img1.size, Image.LANCZOS)
    a1 = cv2.cvtColor(np.array(img1),cv2.COLOR_RGB2GRAY).astype(np.float32)
    a2 = cv2.cvtColor(np.array(img2),cv2.COLOR_RGB2GRAY).astype(np.float32)
    k=(11,11); s=1.5; C1=(0.01*255)**2; C2=(0.03*255)**2
    mu1=cv2.GaussianBlur(a1,k,s); mu2=cv2.GaussianBlur(a2,k,s)
    m1sq=mu1*mu1; m2sq=mu2*mu2; m12=mu1*mu2
    sig1=cv2.GaussianBlur(a1*a1,k,s)-m1sq
    sig2=cv2.GaussianBlur(a2*a2,k,s)-m2sq
    sig12=cv2.GaussianBlur(a1*a2,k,s)-m12
    ssim_map=((2*m12+C1)*(2*sig12+C2))/((m1sq+m2sq+C1)*(sig1+sig2+C2))
    return float(1.0-float(ssim_map.mean()))

# Aesthetic Predictor V2.5
_av25_model=_av25_proc=None
try:
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
    print("Loading Aesthetic Predictor V2.5 …")
    _av25_model,_av25_proc = convert_v2_5_from_siglip(low_cpu_mem_usage=True,trust_remote_code=True)
    _av25_model = _av25_model.to(torch.bfloat16).to(DEVICE_SCORE).eval()
    print("  ✓ Aesthetic V2.5  (SigLIP, 0–10)")
    @torch.inference_mode()
    def _score_av25(img):
        px = _av25_proc(images=img,return_tensors="pt").pixel_values.to(torch.bfloat16).to(DEVICE_SCORE)
        return float(_av25_model(px).logits.squeeze().float().cpu().numpy())
except Exception as e:
    print(f"  [skip] Aesthetic V2.5 — {e}")
    def _score_av25(img): return None

# ShadowAesthetic (anime-trained). v9 fixes the v8 "Unrecognized image processor"
# failure: load the model and processor EXPLICITLY (the repo ships no
# image_processor_type), and pass interpolate_pos_encoding=True so a processor of
# any resolution still works through the ViT. Falls back across mirrors, then to
# the binary cafe_aesthetic only as a last resort.
_shadow_fn=None; _shadow_is_binary=False

def _try_load_shadow_manual(mid: str, hq_label: str):
    from transformers import AutoModelForImageClassification, AutoImageProcessor, ViTImageProcessor
    model = AutoModelForImageClassification.from_pretrained(mid, token=HF_TOKEN).to(DEVICE_SCORE).eval()
    try:
        proc = AutoImageProcessor.from_pretrained(mid, token=HF_TOKEN)
    except Exception:
        proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-384")  # explicit fallback
    id2label = getattr(model.config, "id2label", {0:"hq",1:"lq"}) or {0:"hq",1:"lq"}
    hq_idx = next((int(i) for i,l in id2label.items()
                   if hq_label in str(l).lower() or "high" in str(l).lower()), 0)
    @torch.no_grad()
    def _fn(img):
        inp = proc(images=img, return_tensors="pt").to(DEVICE_SCORE)
        try:    logits = model(**inp, interpolate_pos_encoding=True).logits
        except TypeError: logits = model(**inp).logits
        return float(torch.softmax(logits.float(), -1)[0, hq_idx].cpu())
    return _fn

for _mid,_hq in SHADOW_MODEL_CANDIDATES:
    try:
        print(f"Loading ShadowAesthetic ({_mid}) …")
        _fn=_try_load_shadow_manual(_mid,_hq)
        _v=_fn(source_image)                       # smoke test
        assert _v is not None and 0.0<=_v<=1.0, f"bad score {_v}"
        _shadow_fn=_fn; print(f"  ✓ ShadowAesthetic: {_mid}  (P(hq)={_v:.3f})"); break
    except Exception as _e:
        print(f"  [try] {_mid} — {str(_e)[:120]}")

if _shadow_fn is None:   # last resort: binary cafe_aesthetic (weight suppressed)
    try:
        print("Loading ShadowAesthetic (cafeai/cafe_aesthetic, binary fallback) …")
        _cafe = hf_pipeline("image-classification", model="cafeai/cafe_aesthetic",
                            device=DEVICE_SCORE, dtype=DTYPE)
        def _shadow_fn(img):
            res=_cafe(img)
            for r in res:
                if r["label"].lower()=="aesthetic": return float(r["score"])
            return float(max(res,key=lambda x:x["score"])["score"])
        _shadow_is_binary=True
        print(f"  ⚠  binary fallback — weight reduced to {SHADOW_BINARY_WEIGHT}")
    except Exception as _e:
        print(f"  [skip] shadow — {_e}"); _shadow_fn=None

def _score_shadow(img):
    if _shadow_fn is None: return None
    try:    return _shadow_fn(img)
    except Exception: return None

# Anime aesthetic (skytnt/anime-aesthetic, ONNX) — domain-appropriate primary
# scorer. Self-contained preprocessing (resize-longest-to-768 + pad, NCHW, /255),
# single 0–1 output. Smoke-tested on source + black/white; disabled if degenerate.
_anime_sess=None
if USE_ANIME_AESTHETIC:
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        print(f"Loading anime-aesthetic ({ANIME_AESTHETIC_REPO}) …")
        _anime_path=hf_hub_download(ANIME_AESTHETIC_REPO, ANIME_AESTHETIC_FILE, token=HF_TOKEN)
        _anime_sess=ort.InferenceSession(_anime_path, providers=["CPUExecutionProvider"])
        _anime_inname=_anime_sess.get_inputs()[0].name
    except Exception as e:
        print(f"  [skip] anime-aesthetic load — {e}"); _anime_sess=None

def _score_anime(img: Image.Image):
    if _anime_sess is None: return None
    try:
        arr=np.array(img.convert("RGB"),dtype=np.float32)/255.0
        h,w=arr.shape[:2]; s=768
        if h>w: nh,nw=s,max(1,int(s*w/h))
        else:   nh,nw=max(1,int(s*h/w)),s
        resized=cv2.resize(arr,(nw,nh))
        canvas=np.zeros((s,s,3),dtype=np.float32)
        ph,pw=(s-nh)//2,(s-nw)//2
        canvas[ph:ph+nh,pw:pw+nw]=resized
        inp=canvas.transpose(2,0,1)[np.newaxis,:]          # 1,3,768,768
        pred=_anime_sess.run(None,{_anime_inname:inp})[0]
        return float(np.clip(np.array(pred).reshape(-1)[0],0.0,1.0))
    except Exception:
        return None

if _anime_sess is not None:
    try:
        _a=_score_anime(source_image)
        _aw=_score_anime(Image.new("RGB",(256,256),(255,255,255)))
        _ab=_score_anime(Image.new("RGB",(256,256),(10,10,10)))
        if _a is None or not (0.0<=_a<=1.0) or abs((_aw or 0)-(_ab or 0))<1e-4:
            print(f"  [skip] anime-aesthetic smoke test failed (src={_a}, w={_aw}, b={_ab})")
            _anime_sess=None
        else:
            print(f"  ✓ anime-aesthetic  (source={_a:.3f}, responsive)")
    except Exception as e:
        print(f"  [skip] anime-aesthetic smoke test — {e}"); _anime_sess=None

# NIMA
_nima=None
if USE_PYIQA_NIMA:
    try:
        import pyiqa; print("Loading NIMA …")
        _nima = pyiqa.create_metric("nima",device=DEVICE_SCORE)
        print("  ✓ NIMA  (1–10)")
        def _score_nima(img): return float(_nima(img))
    except Exception as e:
        print(f"  [skip] NIMA — {e}")
        def _score_nima(img): return None
else:
    def _score_nima(img): return None

# MUSIQ
_musiq=None
if USE_PYIQA_MUSIQ:
    try:
        import pyiqa; print("Loading MUSIQ …")
        _musiq = pyiqa.create_metric("musiq",device=DEVICE_SCORE)
        print("  ✓ MUSIQ  (0–100)")
        def _score_musiq(img): return float(_musiq(img))
    except Exception as e:
        print(f"  [skip] MUSIQ — {e}")
        def _score_musiq(img): return None
else:
    def _score_musiq(img): return None

# HELD-OUT validator: scored on every candidate, excluded from composite AND
# from all prompts. Pure guard against overfitting the optimised metrics.
_heldout_metric=None
if USE_HELDOUT_VALIDATION:
    try:
        import pyiqa; print("Loading CLIP-IQA (held-out validator) …")
        _heldout_metric = pyiqa.create_metric("clipiqa",device=DEVICE_SCORE)
        print("  ✓ CLIP-IQA  (0–1, held out of composite + prompts)")
    except Exception as e:
        print(f"  [skip] CLIP-IQA held-out — {e}")
def _score_heldout(img):
    if _heldout_metric is None: return None
    try:    return float(max(0.0, min(1.0, float(_heldout_metric(img)))))
    except Exception: return None

# PickScore — the JUDGE. CLIP-ViT-H fine-tuned on ~1M human preference pairs.
# Loaded on DEVICE_SCORE (~3.9 GB resident); used pairwise to gate rollbacks.
# This is a different model family from the Gemini artist, so it's a genuinely
# independent verdict rather than the model grading its own homework.
_pick_model = _pick_proc = None
if USE_PICKSCORE_JUDGE:
    try:
        from transformers import AutoProcessor
        print("Loading PickScore judge (CLIP-ViT-H) …")
        _pick_proc  = AutoProcessor.from_pretrained(PICKSCORE_CLIP_REPO, token=HF_TOKEN)
        # v10 FIX: load as CLIPModel explicitly. In newer transformers AutoModel
        # resolved PickScore_v1 to a base model whose get_image_features returned
        # a BaseModelOutputWithPooling (not a tensor) → the judge silently died
        # every run since v7. CLIPModel guarantees the dual-tower forward.
        from transformers import CLIPModel
        try:
            _pick_model = CLIPModel.from_pretrained(PICKSCORE_REPO, token=HF_TOKEN).eval().to(DEVICE_SCORE)
        except Exception:
            from transformers import AutoModel
            _pick_model = AutoModel.from_pretrained(PICKSCORE_REPO, token=HF_TOKEN).eval().to(DEVICE_SCORE)
        print("  ✓ PickScore  (human-preference judge, local)")
    except Exception as e:
        print(f"  [skip] PickScore — {e}  (will fall back to Gemini A/B)")
        _pick_model = _pick_proc = None

_pick_err_shown=False
@torch.no_grad()
def _pickscore(img: Image.Image, prompt: str = PICK_PROMPT) -> float | None:
    """PickScore for (img, prompt). Only relative values matter.
    v10: single dual-tower forward → logits_per_image (the canonical PickScore
    computation), which sidesteps the get_image_features return-type bug. Robust
    fallback to manual embeds if a model lacks logits_per_image."""
    global _pick_err_shown
    if _pick_model is None: return None
    try:
        inp = _pick_proc(images=img, text=prompt, return_tensors="pt",
                         padding=True, truncation=True, max_length=77)
        inp = {k: v.to(DEVICE_SCORE) for k, v in inp.items() if hasattr(v, "to")}
        out = _pick_model(**inp)
        lpi = getattr(out, "logits_per_image", None)
        if lpi is not None:
            return float(lpi.reshape(-1)[0].item())
        # fallback: use projected embeds off the same output
        ie = out.image_embeds; te = out.text_embeds
        ie = ie / ie.norm(dim=-1, keepdim=True); te = te / te.norm(dim=-1, keepdim=True)
        return float((_pick_model.logit_scale.exp() * (te @ ie.T)).reshape(-1)[0].item())
    except Exception as e:
        if not _pick_err_shown:
            print(f"  [PickScore] scoring failed — judge DISABLED: {type(e).__name__}: {str(e)[:140]}")
            _pick_err_shown=True
        return None

# Smoke test the judge at load: if it can't score the source, disable it now
# (and the run will fall back to the Gemini A/B judge) instead of silently
# pretending to gate every step like v9 did.
if _pick_model is not None:
    _ps=_pickscore(source_image)
    if _ps is None:
        print("  ⚠ PickScore smoke test FAILED — disabling judge (fallback: Gemini A/B)")
        _pick_model=None
    else:
        print(f"  ✓ PickScore judge live (source raw score {_ps:.3f})")

def _pick_prefers_new(score_new: float | None, score_ckpt: float | None) -> float | None:
    """p(new is preferred over checkpoint) via sigmoid of the score difference.
    None if PickScore unavailable."""
    if score_new is None or score_ckpt is None: return None
    import math
    return 1.0 / (1.0 + math.exp(-(score_new - score_ckpt)))

# LAION V2 — the .pth is a state_dict for this small MLP head (CLIP-L/14 768-d).
class _LaionMLP(torch.nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),        torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),          torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )
    def forward(self, embed):
        return self.layers(embed)

_laion_head=None
try:
    import urllib.request
    _lp = Path(LAION_V2_LOCAL)
    if not _lp.exists():
        print("Downloading LAION V2 from GitHub …")
        _lp.parent.mkdir(parents=True,exist_ok=True)
        _gh_url = (
            "https://raw.githubusercontent.com/christophschuhmann/"
            "improved-aesthetic-predictor/main/ava%2Blogos-l14-linearMSE.pth"
        )
        urllib.request.urlretrieve(_gh_url, str(_lp))
        print(f"  ✓ Downloaded → {_lp}")
    if _lp.exists() and _lp.stat().st_size > 1000:
        print("Loading LAION V2 head …")
        _state = torch.load(str(_lp), map_location=DEVICE_SCORE, weights_only=True)
        if isinstance(_state, torch.nn.Module):
            _laion_head = _state.to(DEVICE_SCORE).eval()
        else:
            _laion_head = _LaionMLP(768)
            _laion_head.load_state_dict(_state)
            _laion_head = _laion_head.to(DEVICE_SCORE).eval()
        print("  ✓ LAION V2  (CLIP-L/14 → MLP head, 0–10)")
        @torch.no_grad()
        def _score_laion(img): return float(_laion_head(clip_embed(img)).squeeze().item())
    else:
        if _lp.exists(): _lp.unlink()
        def _score_laion(img): return None
except Exception as e:
    print(f"  [skip] LAION V2 — {e}")
    def _score_laion(img): return None

_NORM = {
    "anime_aesthetic":  (0.0, 1.0),
    "aesthetic_v25":    (AES_V25_NORM_LO, AES_V25_NORM_HI),
    "shadow_aesthetic": (0.0, 1.0),
    "nima":             (NIMA_NORM_LO, NIMA_NORM_HI),
    "musiq":            (0.0, 100.0),
    "laion_v2":         (0.0, 10.0),
}
def _norm(v,lo,hi): return None if v is None else max(0.0,min(1.0,(v-lo)/(hi-lo)))

# ---------------------------------------------------------------------------
# Weights — dynamic (headroom-proportional) weights are ACTUALLY WIRED UP in v6
# ---------------------------------------------------------------------------
_RUN_WEIGHTS: dict[str, float] = {}

def _effective_base_weights() -> dict:
    base = SCORING_WEIGHTS.copy()
    if _shadow_is_binary:
        base["shadow_aesthetic"] = min(base["shadow_aesthetic"], SHADOW_BINARY_WEIGHT)
    return base

def _compute_run_weights(source_normed: dict) -> dict:
    """Each metric's run-weight ∝ (remaining headroom) × (base weight),
    re-normalised so the sum equals the sum of base weights.
    v10: NO single metric may exceed MAX_METRIC_WEIGHT of the total. In the v9
    run headroom weighting put 0.73 of the weight on the anime scorer alone, so
    the composite WAS that one (gameable) metric and vignette-spam maximised it.
    Capping forces the objective to stay a blend of judges."""
    base = _effective_base_weights()
    if not USE_DYNAMIC_WEIGHTS:
        return base
    hr: dict[str, float] = {}
    for k, w in base.items():
        n = source_normed.get(k)
        hr[k] = 0.0 if n is None else max(0.0, _SCORE_TARGETS.get(k, 0.75) - n) * w
    total_hr = sum(hr.values())
    if total_hr < 1e-6:
        return base
    total_base = sum(base.values())
    dyn = {k: hr[k] / total_hr * total_base for k in base}
    # Cap any single metric at MAX_METRIC_WEIGHT of the total, redistribute the
    # excess to the uncapped metrics (a couple of passes converges).
    cap = MAX_METRIC_WEIGHT * total_base
    for _ in range(4):
        over = {k: v for k, v in dyn.items() if v > cap + 1e-9}
        if not over: break
        excess = sum(v - cap for v in over.values())
        for k in over: dyn[k] = cap
        room = [k for k in dyn if dyn[k] < cap - 1e-9 and hr[k] > 0]
        base_room = sum(hr[k] for k in room) or 1.0
        for k in room: dyn[k] = min(cap, dyn[k] + excess * hr[k] / base_room)
    return dyn

def _composite_from_normed(normed: dict) -> float:
    weights = (_RUN_WEIGHTS or _effective_base_weights())
    tw=0.0; ts=0.0
    for k,w in weights.items():
        n=normed.get(k)
        if n is not None and w>0: ts+=w*n; tw+=w
    return (ts/tw) if tw>0 else 0.0

def _compute_scores(img: Image.Image, drift_ref: "Image.Image | None" = None,
                    semantic: bool = False) -> dict:
    """Score an image.
    struct_drift = SSIM distance vs drift_ref (loose SSIM backstop only).
    cum_drift    = SSIM distance vs source.
    sem_drift / cum_sem_drift = CLIP SEMANTIC distance (the v8 generative gate),
    computed only when semantic=True (generative candidates) — parametric edits
    are drift-exempt so we skip the extra CLIP passes for them."""
    raw = {
        "anime_aesthetic":  _score_anime(img),
        "aesthetic_v25":    _score_av25(img),
        "shadow_aesthetic": _score_shadow(img),
        "nima":             _score_nima(img),
        "musiq":            _score_musiq(img),
        "laion_v2":         _score_laion(img),
    }
    normed = {k: _norm(raw[k],*_NORM[k]) for k in _NORM}
    composite = _composite_from_normed(normed)
    cum_drift = _ssim_drift(source_image, img)
    struct_drift = cum_drift if drift_ref is None else _ssim_drift(drift_ref, img)
    if semantic:
        cum_sem = _clip_drift(source_image, img)
        sem = cum_sem if drift_ref is None else _clip_drift(drift_ref, img)
    else:
        cum_sem = sem = None
    heldout = _score_heldout(img)
    return {
        "anime_aesthetic":  round(raw["anime_aesthetic"],3)  if raw["anime_aesthetic"]  is not None else None,
        "aesthetic_v25":    round(raw["aesthetic_v25"],3)    if raw["aesthetic_v25"]    is not None else None,
        "shadow_aesthetic": round(raw["shadow_aesthetic"],3) if raw["shadow_aesthetic"] is not None else None,
        "nima":             round(raw["nima"],3)             if raw["nima"]             is not None else None,
        "musiq":            round(raw["musiq"],2)            if raw["musiq"]            is not None else None,
        "laion_v2":         round(raw["laion_v2"],3)         if raw["laion_v2"]         is not None else None,
        "struct_drift":     round(struct_drift,4),
        "cum_drift":        round(cum_drift,4),
        "sem_drift":        round(sem,4) if sem is not None else None,
        "cum_sem_drift":    round(cum_sem,4) if cum_sem is not None else None,
        "heldout":          round(heldout,4) if heldout is not None else None,
        "composite":        round(composite,4),
        "normed":           {k:round(v,4) for k,v in normed.items() if v is not None},
    }

print("\nComputing source scores …")
SOURCE_SCORES = _compute_scores(source_image)
# Wire up dynamic weights (dead code in v5) and recompute the source composite
# under them so every Δcomposite this run compares like with like.
_RUN_WEIGHTS = _compute_run_weights(SOURCE_SCORES["normed"])
SOURCE_SCORES["composite"] = round(_composite_from_normed(SOURCE_SCORES["normed"]), 4)
print("  Source:")
for k,v in SOURCE_SCORES.items():
    if k!="normed": print(f"    {k:<22} {v}")
print("  Normalised (0→1):")
for k,v in SOURCE_SCORES["normed"].items(): print(f"    {k:<22} {v:.4f}")
print("  Run weights (headroom-proportional):")
for k,w in sorted(_RUN_WEIGHTS.items(), key=lambda x:-x[1]):
    print(f"    {k:<22} {w:.4f}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Generative backend: CosXL Edit preferred, InstructPix2Pix fallback. Both expose
# the same InstructPix2Pix sampling interface (image + instruction +
# image_guidance_scale + guidance_scale), so run_generative_edit is backend-blind.
_gen_pipe = None
_gen_backend = None   # "cosxl" | "ip2p" | None
_generative_needed = any(v=="GENERATIVE" for v in EDIT_CATEGORIES.values())

def _load_cosxl():
    from huggingface_hub import hf_hub_download
    from diffusers import StableDiffusionXLInstructPix2PixPipeline, EDMEulerScheduler
    print(f"Loading CosXL Edit ({COSXL_REPO}) …")
    edit_file = hf_hub_download(repo_id=COSXL_REPO, filename=COSXL_FILE, token=HF_TOKEN)
    pipe = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
        edit_file, num_in_channels=8, is_cosxl_edit=True, torch_dtype=DTYPE)
    pipe.scheduler = EDMEulerScheduler(
        sigma_min=0.002, sigma_max=120.0, sigma_data=1.0,
        prediction_type="v_prediction", sigma_schedule="exponential")
    pipe.to(DEVICE_EDIT)
    try: pipe.enable_vae_slicing(); pipe.enable_vae_tiling()
    except Exception: pass
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    return pipe

def _load_ip2p():
    from diffusers import StableDiffusionInstructPix2PixPipeline
    print("Loading InstructPix2Pix …")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=DTYPE, safety_checker=None).to(DEVICE_EDIT)
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    return pipe

if _generative_needed:
    if GENERATIVE_BACKEND in ("cosxl",):
        try:
            _gen_pipe = _load_cosxl(); _gen_backend = "cosxl"
            print(f"  ✓ CosXL Edit on {DEVICE_EDIT}  (SDXL instruction editing)")
        except Exception as e:
            print(f"  [warn] CosXL Edit failed to load ({e}) — falling back to InstructPix2Pix")
    if _gen_backend is None:
        try:
            _gen_pipe = _load_ip2p(); _gen_backend = "ip2p"
            print(f"  ✓ InstructPix2Pix on {DEVICE_EDIT}  (fallback backend)")
        except Exception as e:
            print(f"  [skip] no generative backend available ({e}) — GENERATIVE edits disabled")
else:
    print("No GENERATIVE edits requested — generative backend not loaded.")

_GENERATIVE_FALLBACK_INSTRUCTIONS: dict[str,str] = {
    "style_transfer": "Subtly shift the artistic style of the photo while keeping the subject, composition and lighting recognisable",
    "color_grade":    "Apply a cinematic colour grade to the photo, adjusting the overall colour balance and mood",
}

def _fallback_ip2p_instruction(category: str, strength: float) -> str:
    base = _GENERATIVE_FALLBACK_INSTRUCTIONS.get(
        category, f"Adjust the {category.replace('_',' ')} of the photo")
    return base + (" — apply it boldly." if abs(strength)>0.6 else " — keep the change subtle.")

def _gen_scales(strength: float) -> tuple[float, float]:
    """Map |strength| in (0,1] to (image_guidance_scale, guidance_scale) for the
    active backend."""
    s = max(0.05, min(1.0, abs(strength)))
    if _gen_backend == "cosxl":
        ig_lo, ig_hi = COSXL_IMAGE_GUIDANCE_RANGE; tg_lo, tg_hi = COSXL_TEXT_GUIDANCE_RANGE
    else:
        ig_lo, ig_hi = IP2P_IMAGE_GUIDANCE_RANGE;  tg_lo, tg_hi = IP2P_TEXT_GUIDANCE_RANGE
    return round(ig_lo + s*(ig_hi-ig_lo),2), round(tg_lo + s*(tg_hi-tg_lo),2)

def _fit_for_diffusion(img: Image.Image, target_long: int):
    """Resize so the long edge is target_long and both dims are multiples of 8
    (SDXL/SD requirement). Returns (resized, original_size)."""
    w, h = img.size
    scale = target_long / max(w, h)
    nw = max(8, int(round(w * scale / 8)) * 8)
    nh = max(8, int(round(h * scale / 8)) * 8)
    return img.resize((nw, nh), Image.LANCZOS), (w, h)

def run_generative_edit(img: Image.Image, instruction: str, strength: float = 0.5) -> Image.Image:
    if _gen_backend is None: raise RuntimeError("No generative backend available.")
    img_cfg, txt_cfg = _gen_scales(strength)
    # CosXL/SDXL wants ~768–1024; edit at EDIT_RESOLUTION then resize back so the
    # scored/saved image stays at the source resolution.
    work, orig_size = _fit_for_diffusion(img, EDIT_RESOLUTION)
    steps = COSXL_NUM_INFERENCE_STEPS if _gen_backend == "cosxl" else IP2P_NUM_INFERENCE_STEPS
    kwargs = dict(prompt=instruction, image=work, num_inference_steps=steps,
                  image_guidance_scale=img_cfg, guidance_scale=txt_cfg)
    if _gen_backend == "cosxl":
        kwargs.update(height=work.size[1], width=work.size[0], negative_prompt="")
    out = _gen_pipe(**kwargs).images[0]
    if out.size != orig_size:
        out = out.resize(orig_size, Image.LANCZOS)
    return out

# %% [code] {"jupyter":{"outputs_hidden":false}}
def _image_to_b64(img: Image.Image) -> str:
    buf=io.BytesIO(); img.save(buf,format="JPEG",quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def _build_tool_menu() -> str:
    lines=[]
    for cat,desc in EDIT_TOOL_DESCRIPTIONS.items():
        tag="[P]" if EDIT_CATEGORIES.get(cat)=="PARAMETRIC" else "[G]"
        lines.append(f"  {tag} {cat:<26} {desc}")
    return "\n".join(lines)

def _detect_cycle(history: list, window: int=6, threshold: int=3) -> str:
    recent=[h for h in history[-window:] if h.get("accepted")]
    if not recent: return ""
    counts=Counter(h["category"] for h in recent)
    overused=[f"{c}(×{n})" for c,n in counts.items() if n>=threshold]
    if overused:
        return (f"\n⚠ LOOP: {', '.join(overused)} repeated {threshold}+ times in {window} steps."
                " Break out — try a different category or [G] generative tool.")
    return ""

# v8: judges speak English. Each scorer maps to a plain-language name and band
# so the artist reasons over a reading, not a table of normalised floats.
# Each judge: a plain name, what it actually measures, and whether it was trained
# on PHOTOS (so it sits mid-scale on anime and shouldn't be chased to a perfect
# number) or on ANIME (trust it most for this material).
_METRIC_INFO = {
    "anime_aesthetic":  ("anime aesthetic judge", "trained on ANIME art — trust this one most here", "anime"),
    "shadow_aesthetic": ("illustration-quality judge", "trained on ANIME/illustration", "anime"),
    "aesthetic_v25":    ("general aesthetic appeal", "rewards clean light, depth, colour harmony", "photo"),
    "musiq":            ("perceived sharpness & clarity", "responds to detail and crispness", "photo"),
    "nima":             ("photographic quality (mean opinion)", "a 1–10 mean-opinion score for exposure/focus/comp", "photo"),
    "laion_v2":         ("trained aesthetic taste", "a generic web-image taste model", "photo"),
}
def _metric_name(k): return _METRIC_INFO.get(k,(k,"",""))[0]
def _band(n: float) -> str:
    if n < 0.35: return "very weak"
    if n < 0.50: return "weak"
    if n < 0.65: return "middling"
    if n < 0.80: return "good"
    return "excellent"
def _trend(delta: float) -> str:
    if delta >  0.02: return "much better than the original"
    if delta >  0.005: return "a little better than the original"
    if delta < -0.02: return "much worse than the original"
    if delta < -0.005: return "slightly worse than the original"
    return "about the same as the original"

def _last_accepted_scores(history: list) -> dict | None:
    for h in reversed(history or []):
        if h.get("accepted") and isinstance(h.get("scores"), dict):
            return h["scores"]
    return None

def _build_score_panel(scores: dict, source_scores: dict, history: list | None=None) -> str:
    normed     = scores.get("normed",{})
    src_normed = source_scores.get("normed",{})
    weights    = (_RUN_WEIGHTS or _effective_base_weights())
    ranked=[]
    for k,w in weights.items():
        n=normed.get(k)
        if n is None: continue
        ranked.append((max(0.0,_SCORE_TARGETS.get(k,0.75)-n)*w, k, n))
    ranked.sort(reverse=True)
    has_anime = any(_METRIC_INFO.get(k,("","",""))[2]=="anime" for _,k,_ in ranked)

    lines=[
      "WHERE THE IMAGE STANDS — an art-director's reading of the automated judges.",
      "Most of these critics were trained on PHOTOGRAPHS, so on a stylised anime",
      "frame they sit mid-scale and rarely go high — they are a rough compass, not",
      "a target to max out. Trust the ANIME-trained judges and your own eyes most.",
      ""]
    for i,(pri,k,n) in enumerate(ranked):
        src_n=src_normed.get(k); delta=(n-src_n) if src_n is not None else 0.0
        _,what,dom=_METRIC_INFO.get(k,(k,"",""))
        anime_tag="  [ANIME-trained — weigh heavily]" if dom=="anime" else ""
        opp=" ← clearest opportunity" if i==0 and n<_SCORE_TARGETS.get(k,0.75) else (
            " (already strong)" if n>=_SCORE_TARGETS.get(k,0.75) else "")
        lines.append(f"  • {_metric_name(k)}: {_band(n)}, {_trend(delta)}.{opp}{anime_tag}")
        lines.append(f"      ({what})")

    # Held-out advisory (CLIP-IQA) — shown, never a gate in v9.
    hc=scores.get("heldout")
    if hc is not None:
        lines.append(f"  • technical-cleanliness check (CLIP-IQA, advisory only): {_band(hc)} "
                     f"— watch for over-sharpening/over-saturation, but a deliberate bold look is fine.")

    c=scores.get("composite",0); sc=source_scores.get("composite",0); dc=c-sc
    overall = ("an improvement on the original" if dc>0.01 else
               "essentially unchanged from the original" if dc>-0.01 else
               "currently a step down from the original")
    lines.append(f"  Overall standing: {overall}.")

    # What changed since the LAST accepted edit, in words.
    prev=_last_accepted_scores(history)
    if prev:
        moved=[]
        for _,k,n in ranked:
            pn=(prev.get("normed") or {}).get(k)
            if pn is None: continue
            d=n-pn
            if abs(d)>=0.01:
                moved.append(f"{_metric_name(k)} {'rose' if d>0 else 'fell'}")
        if moved:
            lines.append("  Since your last accepted edit: " + "; ".join(moved[:4]) + ".")
        else:
            lines.append("  Since your last accepted edit: little has moved — try a bolder direction.")

    cum_sem=scores.get("cum_sem_drift")
    if cum_sem is not None:
        room = ("plenty of room to keep transforming it" if cum_sem<0.25 else
                "getting far from the original — keep further changes purposeful" if cum_sem<CUM_SEM_DRIFT_CAP
                else "near the limit of how far it may depart from the source")
        lines.append(f"  You still have {room}.")

    if history:
        sc_src=source_scores.get("composite") or 0
        accepted=[h for h in history if h.get("accepted") and isinstance(h.get("scores"),dict)]
        beat=any(((h.get("scores") or {}).get("composite") or 0)-sc_src>0.01 for h in accepted)
        ptweaks=sum(1 for h in accepted if EDIT_CATEGORIES.get(h.get("category",""))=="PARAMETRIC")
        if not beat and ptweaks>=3:
            lines.append("  NOTE: small parametric tweaks have barely moved the needle — this is the moment "
                         "for a BOLD generative move (recolour / relight / restyle), not another micro-edit.")
    lines.append(f"  (numeric footnote — composite {c:.3f} vs source {sc:.3f}; "
                 f"{', '.join(f'{_metric_name(k).split()[0]} {n:.2f}' for _,k,n in ranked)})")
    return "\n".join(lines)

def _build_history_summary(history: list, n: int=8) -> str:
    if not history: return "  (no history)"
    recent=history[-n:]
    lines=[f"  Last {len(recent)} steps:"]
    for h in recent:
        gate="✓" if h.get("accepted") else "✗"
        c=(h.get("scores") or {}).get("composite")
        cs=f"composite={c:.4f}" if c is not None else ""
        rsn="" if h.get("accepted") else f"  ← {h.get('reason','')}"
        nc=h.get("n_candidates",1)
        ncs=f"  (best of {nc})" if nc and nc>1 else ""
        lines.append(f"    {gate} step{h.get('step','?'):02d}  {h.get('category','?'):<26} "
                     f"{h.get('strength',0):+.2f}  [{h.get('intent','?')}]  {cs}{rsn}{ncs}")
    cats=Counter(h["category"] for h in history if h.get("accepted"))
    if cats: lines.append(f"  Most accepted: {', '.join(f'{c}(×{n})' for c,n in cats.most_common(3))}")
    lines.append("  Category avg Δcomposite (accepted):")
    sc_src=SOURCE_SCORES.get("composite") or 0
    cat_d: dict[str,list[float]]={}
    for h in history:
        if not h.get("accepted"): continue
        cv=(h.get("scores") or {}).get("composite")
        if cv is None: continue
        cat_d.setdefault(h["category"],[]).append(cv-sc_src)
    for cat,ds in sorted(cat_d.items(),key=lambda x:-sum(x[1])/len(x[1])):
        avg=sum(ds)/len(ds)
        lines.append(f"    {cat:<26} avg={avg:+.4f} ({len(ds)} step(s))")
    return "\n".join(lines)

# --- Empirical "understanding of the editor" — now DIRECTION-AWARE -------------
def _dir_key(category: str, strength: float) -> str:
    return f"{category}[{'+' if (strength or 0)>=0 else '-'}]"

def _tool_profile_dict(attempt_log: list) -> dict:
    """Per-(category,direction), per-metric NORMALISED Δ across EVERY attempt
    (accepted or not). v6: keyed by direction — v5 averaged contrast(+0.5) and
    contrast(-0.5) into one meaningless number — and also tracks Δcomposite,
    which powers category blocking."""
    agg: dict[str, dict[str, list[float]]] = {}
    for a in attempt_log:
        cat = a.get("category")
        if not cat: continue
        key  = _dir_key(cat, a.get("strength", 0.0))
        cur  = (a.get("scores") or {})
        prev = (a.get("prev_scores") or {})
        cn, pn = cur.get("normed", {}), prev.get("normed", {})
        for k, v in cn.items():
            pv = pn.get(k)
            if pv is None: continue
            agg.setdefault(key, {}).setdefault(k, []).append(round(v-pv, 5))
        cc, pc = cur.get("composite"), prev.get("composite")
        if cc is not None and pc is not None:
            agg.setdefault(key, {}).setdefault("composite", []).append(round(cc-pc, 5))
    return agg

def _build_tool_profile(profile: dict, prior: dict | None = None, min_n: int = TOOL_PROFILE_MIN_N) -> str:
    if not profile and not prior:
        return "TOOL EFFECT PROFILE\n  (no attempts yet — this fills in as edits are tried)"
    lines=["TOOL EFFECT PROFILE  (empirical normalised Δ per metric, by tool[direction]; + helps, − hurts; n=attempts)"]
    cats = sorted(set((profile or {}).keys()) | set((prior or {}).keys()))
    any_row=False
    for cat in cats:
        this = (profile or {}).get(cat, {})
        pri  = (prior or {}).get(cat, {})
        parts=[]
        for k in sorted(set(this) | set(pri)):
            seg=[]
            if k in this and len(this[k]) >= min_n:
                avg = sum(this[k])/len(this[k])
                seg.append(f"{k}{avg:+.3f}(n={len(this[k])})")
            if k in pri and len(pri[k]) >= min_n:
                avgp = sum(pri[k])/len(pri[k])
                seg.append(f"[prior {k}{avgp:+.3f},n={len(pri[k])}]")
            if seg: parts.append(" ".join(seg))
        if parts:
            lines.append(f"  {cat:<24} " + "  ".join(parts))
            any_row=True
    if not any_row: lines.append("  (not enough attempts yet)")
    return "\n".join(lines)

def _build_calibration(attempt_log: list) -> str:
    """Did proposals that set 'targeting=X' actually move metric X up?"""
    stats: dict[str, list[bool]] = {}
    for a in attempt_log:
        tgt  = a.get("targeting")
        cur  = (a.get("scores") or {}).get("normed", {})
        prev = (a.get("prev_scores") or {}).get("normed", {})
        if not tgt or tgt not in cur or tgt not in prev: continue
        stats.setdefault(tgt, []).append(cur[tgt] > prev[tgt])
    if not stats: return ""
    parts=[]
    for k,hits in stats.items():
        n=len(hits)
        parts.append(f"{k}: {sum(hits)}/{n} hit ({sum(hits)/n:.0%})")
    return "CALIBRATION (did 'targeting=X' proposals actually raise X?)\n  " + "  ·  ".join(parts)

def _merge_profiles(a: dict, b: dict) -> dict:
    out = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in (a or {}).items()}
    for cat, metrics in (b or {}).items():
        for k, ds in metrics.items():
            out.setdefault(cat, {}).setdefault(k, []).extend(ds)
    return out

# --- Category blocking (config existed in v5; implemented in v6) ---------------
def _blocked_set(merged_profile: dict) -> set:
    """(tool_key, metric) pairs with ≥ MIN_N attempts whose average Δ is below
    BLOCKED_CATEGORY_THRESHOLD. tool_key is direction-aware ('contrast[+]')
    for v6 data, plain ('contrast') for profiles carried over from v5."""
    blocked=set()
    for cat, metrics in (merged_profile or {}).items():
        for k, ds in metrics.items():
            if len(ds) >= BLOCKED_CATEGORY_MIN_N and sum(ds)/len(ds) < BLOCKED_CATEGORY_THRESHOLD:
                blocked.add((cat, k))
    return blocked

def _is_blocked(cand: dict, blocked: set) -> bool:
    cat = cand.get("category"); s = cand.get("strength", 0.0)
    tgt = cand.get("targeting") or ""
    for key in (_dir_key(cat, s), cat):
        if (key, "composite") in blocked: return True
        if tgt and (key, tgt) in blocked: return True
    return False

def _build_blocked_block(blocked: set) -> str:
    if not blocked: return ""
    items = sorted(f"{c}→{m}" for c, m in blocked)
    return ("\nHARD-BLOCKED (empirically net-negative, will be filtered if proposed): "
            + ", ".join(items[:20]))

def _exploration_floor(step: int) -> float:
    """Cooling schedule for the composite-drop acceptance floor."""
    t = min(1.0, step / max(1, MAX_EDIT_STEPS-1))
    return EXPLORATION_FLOOR_START + t*(EXPLORATION_FLOOR_END-EXPLORATION_FLOOR_START)

def _targets_met(scores: dict) -> bool:
    normed = scores.get("normed", {})
    weights = (_RUN_WEIGHTS or _effective_base_weights())
    rel = [k for k, w in weights.items() if w > 0.01 and normed.get(k) is not None]
    return bool(rel) and all(normed[k] >= _SCORE_TARGETS.get(k, 0.75) for k in rel)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ---------------------------------------------------------------------------
# LLM plumbing — plain REST for BOTH providers (this is what makes the key
# rotator actually work) with native JSON mode on structured calls.
# ---------------------------------------------------------------------------
def _img_part(img: Image.Image) -> dict:
    return {"inlineData":{"mimeType":"image/jpeg","data":_image_to_b64(img)}}

# --- Model roster: demote a model on error, fall to the next-best -------------
class ModelRoster:
    """Ranked model fallback that LEARNS at runtime (v10). The v9 run wasted
    minutes because the auto-ranked premium models were 429-only on the free tier
    and got retried first on every step. v10: a model that fails gets an
    ESCALATING bench (90s → 180s → … capped), and available() floats models that
    have actually SUCCEEDED to the front — so after step 0 the working model is
    tried first and the dead ones sink."""
    def __init__(self, ids: list[str], cooldown: int):
        self.ids=list(ids); self.cooldown=cooldown
        self.block_until={m:0.0 for m in self.ids}
        self.uses=Counter(); self.fails=Counter(); self.streak={m:0 for m in self.ids}
    def available(self) -> list[str]:
        now=time.monotonic()
        av=[m for m in self.ids if self.block_until.get(m,0.0)<=now]
        if not av: av=list(self.ids)                 # all benched? try anyway
        # proven-good models (uses>0) first, then original rank
        av.sort(key=lambda m: (0 if self.uses[m]>0 else 1, self.ids.index(m)))
        return av
    def demote(self, m: str):
        self.fails[m]+=1; self.streak[m]=self.streak.get(m,0)+1
        self.block_until[m]=time.monotonic()+min(self.cooldown*(2**(self.streak[m]-1)), 1800)
    def use(self, m: str): self.uses[m]+=1; self.streak[m]=0
    def report(self) -> list[str]:
        return [f"{m.split('/')[-1]}: {self.uses[m]} ok / {self.fails[m]} fail"
                for m in self.ids if self.uses[m] or self.fails[m]]

gemini_roster = ModelRoster(GEMINI_ROSTER, MODEL_COOLDOWN_SECONDS)
groq_roster   = ModelRoster(GROQ_ROSTER,   MODEL_COOLDOWN_SECONDS)
LAST_MODELS: dict = {"gemini": None, "groq": None}

# v10 fast caller: tries each model over the raw key list WITHOUT globally
# blocking a key on 429 (Gemini quota is per-model-per-key, so one model's 429
# must not poison the key for another model — the v9 bug that caused 64s waits).
# A model that 429s on all keys is demoted and we move on immediately; only if
# EVERY model is rate-limited do we wait once (bounded) and retry.
def _call_models(roster: ModelRoster, keys: list[str], which: str,
                 raw_fn, parse=None):
    deadline=time.monotonic()+180; wait=4.0
    while time.monotonic()<deadline:
        any_rate_limited=False; last=None
        for model in roster.available():
            rl=0
            for key in keys:
                try:
                    out=raw_fn(key, model)
                    if parse is not None: out=parse(out)
                    roster.use(model); LAST_MODELS[which]=model; return out
                except requests.HTTPError as e:
                    code=e.response.status_code if e.response is not None else 0
                    if code in (429,403): rl+=1; last=e; continue      # next key
                    last=e; break                                       # other HTTP: next model
                except (requests.ConnectionError, requests.Timeout) as e:
                    last=e; continue
                except Exception as e:                                   # parse/empty: next model
                    last=e; break
            if rl==len(keys): any_rate_limited=True
            roster.demote(model)
        if not any_rate_limited:
            raise RuntimeError(f"All {which} models failed. Last: {last}")
        print(f"    [{which}] all models rate-limited — waiting {wait:.0f}s …")
        time.sleep(wait); wait=min(wait*2, RPM_COOLDOWN)
    raise RuntimeError(f"All {which} models rate-limited for 180s.")

def _loads_lenient(text: str) -> dict:
    """Parse possibly-truncated/fenced LLM JSON. Repairs an unterminated string
    and unbalanced braces/brackets (the v7 'Unterminated string' truncation)."""
    text=text.strip()
    if "```" in text:
        for part in text.split("```"):
            part=part.strip().lstrip("json").strip()
            if part.startswith("{"): text=part; break
    s=text.find("{")
    if s!=-1: text=text[s:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # repair: balance quotes, then close open brackets/braces in stack order
    t=text
    if t.count('"')%2==1: t+='"'
    stack=[]
    for ch in t:
        if ch in '{[': stack.append('}' if ch=='{' else ']')
        elif ch in '}]':
            if stack and stack[-1]==ch: stack.pop()
    t=t.rstrip().rstrip(',')
    t+="".join(reversed(stack))
    return json.loads(t)            # may still raise — caller treats as failure

# --- Gemini over REST, with model fallback ------------------------------------
def _gemini_raw(api_key: str, model: str, parts: list, temperature: float,
                max_tokens: int, json_mode: bool) -> str:
    url=f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={api_key}"
    gen={"temperature":temperature,"maxOutputTokens":max_tokens}
    if json_mode: gen["responseMimeType"]="application/json"
    resp=requests.post(url, json={"contents":[{"parts":parts}],"generationConfig":gen}, timeout=120)
    resp.raise_for_status()
    data=resp.json()
    cand=(data.get("candidates") or [{}])[0]
    out="".join(p.get("text","") for p in cand.get("content",{}).get("parts",[]))
    if not out.strip():
        fr=cand.get("finishReason", data.get("promptFeedback","?"))
        raise RuntimeError(f"empty text (finishReason={fr})")
    return out

def _gemini_text(parts: list, *, temperature: float, max_tokens: int) -> str:
    return _call_models(gemini_roster, GOOGLE_KEYS, "gemini",
                        lambda k,m: _gemini_raw(k,m,parts,temperature,max_tokens,False),
                        parse=lambda s: s.strip())

def _gemini_json(parts: list, *, temperature: float, max_tokens: int) -> dict:
    return _call_models(gemini_roster, GOOGLE_KEYS, "gemini",
                        lambda k,m: _gemini_raw(k,m,parts,temperature,max_tokens,True),
                        parse=_loads_lenient)

# --- Groq over REST, with model fallback --------------------------------------
def _groq_raw(api_key: str, model: str, prompt: str, temperature: float,
              max_tokens: int, json_mode: bool) -> str:
    payload={"model":model,"messages":[{"role":"user","content":prompt}],
             "max_tokens":max_tokens,"temperature":temperature}
    if json_mode: payload["response_format"]={"type":"json_object"}
    resp=requests.post("https://api.groq.com/openai/v1/chat/completions",
                       headers={"Authorization":f"Bearer {api_key}"}, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def _groq_text(prompt: str, *, temperature: float, max_tokens: int) -> str:
    return _call_models(groq_roster, GROQ_KEYS, "groq",
                        lambda k,m: _groq_raw(k,m,prompt,temperature,max_tokens,False))

def _groq_json(prompt: str, *, temperature: float, max_tokens: int) -> dict:
    return _call_models(groq_roster, GROQ_KEYS, "groq",
                        lambda k,m: _groq_raw(k,m,prompt,temperature,max_tokens,True),
                        parse=_loads_lenient)

def _gemini_describe_scene(img: Image.Image, is_update: bool=False) -> str:
    moment="after edits" if is_update else "before any edits"
    prompt=f"""You are an art director analysing an image {moment}.
Write a concise aesthetic profile (150 words max):
1. SUBJECT & COMPOSITION  2. TONAL RANGE  3. COLOUR PALETTE
4. TEXTURE & DETAIL  5. MOOD  6. WEAKEST ATTRIBUTE
Be specific. Name concrete visual elements."""
    return _gemini_text([{"text":prompt}, _img_part(img)],
                        temperature=TEMP_DESCRIBE, max_tokens=2048)

def _gemini_propose(img, scores, history, attempt_log, scene_desc,
                    critic_note=None, blocked: set | None = None):
    role="FINALIZE your edit decision" if critic_note else "PROPOSE your next edit(s)"
    critic_block=""
    if critic_note:
        critic_block=f"""
━━ CRITIC RESPONSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{critic_note}
You may: AGREE (primary stands) / MODIFY (same category, different strength) /
REDIRECT (different category). Justify your choice in your rationale, and
return the full "candidates" list again (siblings may stay unchanged).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    prompt=f"""You are a vision artist in a digital darkroom. {role}.

The image attached to THIS message is the CURRENT state of the photo (after every
edit accepted so far) — judge it with your own eyes, not from memory. The scene
reference below describes the ORIGINAL only, as a fixed anchor for what the photo
fundamentally is; it is NOT the current state.

━━ ORIGINAL SCENE ANCHOR (not the current image) ━━━━━━━━━━━━━━━━━━━━━━
{scene_desc}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━ SCORE PANEL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{_build_score_panel(scores,SOURCE_SCORES,history)}
{_detect_cycle(history)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━ TOOL EFFECT PROFILE & CALIBRATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{_build_tool_profile(_tool_profile_dict(attempt_log), PRIOR_TOOL_PROFILE)}

{_build_calibration(attempt_log) or "  (no targeting history yet)"}
{_build_blocked_block(blocked or set())}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━ EDIT TOOLS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[P]=parametric (precise, cheap — every candidate AND automatic ×0.65/×1.45
    strength variants are executed and scored; the best accepted one is kept)
[G]=generative (a real re-paint of the image — recolour, relight, restyle.
    Colour/tone/mood are FREE to change; only turning it into a different
    picture is blocked. Use these whenever you want a bold, transformative look.)
{_build_tool_menu()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━ HISTORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{_build_history_summary(history)}
{critic_block}
YOU ARE AN ARTIST, NOT A CALIBRATION TOOL. Tiny safe tweaks are not the goal —
a beautiful, intentional image is. You are FREE to reinvent the colour palette,
the lighting and the mood. Be decisive and bold when the image calls for it.

ABOUT THE SCORES: most of the judges above were trained on PHOTOGRAPHS, so on a
stylised anime frame they sit mid-scale (~0.5) by their very nature and will
NEVER reach a perfect score no matter how good the art looks. Do NOT chase a high
number — it is a mirage. Your eyes and the ANIME-trained judges are the real
signal; the photo scores are only a rough compass.

INSTRUCTIONS:
0. YOUR OWN VERDICT FIRST. Look at the attached current image and, in "see",
   write one honest sentence on what you actually SEE — its biggest strength and
   its biggest weakness right now — BEFORE you look at any number. Then form an
   artistic INTENTION for where to take it.
1. Now reconcile your eyes with the score reading. Where they agree, act with
   confidence; where a photo-trained number disagrees with your eyes, TRUST YOUR
   EYES. Decide what would most improve the IMAGE as a piece of art.
2. If a global colour/light/mood change is what's needed, reach for a [G]
   GENERATIVE tool (style_transfer / color_grade) — these are encouraged, not a
   last resort. Set "ip2p_instruction": a short, concrete, IMPERATIVE sentence
   describing ONE transformation, e.g. "Bathe the scene in warm golden-hour light
   with deep amber shadows" or "Give it a cool, melancholic teal-and-blue night
   palette". Describe the LOOK you want; do not name tools or numbers.
3. If the fix is local or tonal, use [P] PARAMETRIC: list 2-3 DIFFERENT
   candidates (categories and/or strengths) — competing hypotheses, all tested
   automatically, best kept. Prefer the LOCAL tools (subject_clarity /
   background_recede / subject_glow) when the problem is in one REGION.
4. Choose strengths that MATTER — favour decisive values (|s| ≥ 0.3 for a real
   change); never near ±0.05.
5. Do NOT propose anything in the HARD-BLOCKED list, and avoid any
   category+direction the TOOL EFFECT PROFILE shows as net-negative.
6. The ONE rule on boldness: a human-preference judge compares each result to the
   last approved state. Bold and beautiful is rewarded; garish oversharpening /
   oversaturation / fake-HDR is rolled back. Aim for striking, not gaudy.
7. Set "primary_index" to your single best candidate.
8. Keep "see" to ONE sentence and "why" to ≤12 words. Be terse — long text risks
   getting cut off.

Respond ONLY with compact JSON (no fences, no preamble):
{{"see":"<one sentence: what you SEE — strength & weakness>","primary_index":0,"candidates":[{{"category":"<tool>","strength":<-1.0..1.0>,"why":"<≤12 words>","targeting":"<metric>","intent":"<one word>","ip2p_instruction":null}}]}}"""
    return _sanitize_candidates(_gemini_json([{"text":prompt}, _img_part(img)],
                                temperature=TEMP_ARTIST, max_tokens=GEMINI_MAX_OUTPUT_TOKENS))

def _groq_critique(proposal, scores, history, attempt_log, scene_desc):
    cands = proposal.get("candidates", [])
    pidx  = proposal.get("primary_index", 0)
    primary  = cands[pidx] if cands else {}
    siblings = [c for i,c in enumerate(cands) if i != pidx]
    sib_block = (f"\nALSO QUEUED (tested empirically regardless, if [P] PARAMETRIC):\n"
                  f"{json.dumps(siblings,indent=2)}\n") if siblings else ""
    prompt=f"""You are an analytical critic advising a vision artist. You cannot see the
image, but you receive the SAME disaggregated per-metric score panel the artist
sees — composite is shown too, but treat the per-metric breakdown, the TOOL
EFFECT PROFILE, and CALIBRATION as your primary evidence.

ARTIST'S PRIMARY CHOICE (this is what you are critiquing):
{json.dumps(primary,indent=2)}
{sib_block}
SCENE: {scene_desc[:350]}

SCORES:
{_build_score_panel(scores,SOURCE_SCORES,history)}
{_detect_cycle(history)}

{_build_tool_profile(_tool_profile_dict(attempt_log), PRIOR_TOOL_PROFILE)}

{_build_calibration(attempt_log) or "  (no targeting history yet)"}

HISTORY:
{_build_history_summary(history)}

Assess the PRIMARY choice: (1) does it target the [1] priority metric (or is
there a good reason not to)? (2) is ±{primary.get('strength',0)} the right
magnitude? (3) does the TOOL EFFECT PROFILE show this category+direction as
already net-negative for the targeted metric?

Start with exactly one of:
  AGREE    — [one sentence why it's well-targeted]
  MODIFY   — [same category, specific strength, reason]
  REDIRECT — [different category + strength + reason]
Then 1-2 sentences of analysis."""
    return _groq_text(prompt, temperature=TEMP_CRITIC, max_tokens=400)

def _groq_decide_alone(scores, history, attempt_log, scene_desc, blocked: set | None = None):
    prompt=f"""Vision unavailable. Decide from scores only.
SCENE: {scene_desc[:250]}

SCORES:
{_build_score_panel(scores,SOURCE_SCORES,history)}

{_build_tool_profile(_tool_profile_dict(attempt_log), PRIOR_TOOL_PROFILE)}

{_build_calibration(attempt_log) or "  (no targeting history yet)"}
{_build_blocked_block(blocked or set())}

TOOLS:
{_build_tool_menu()}

HISTORY:
{_build_history_summary(history)}

Since you cannot see the image, propose 2-3 [P] PARAMETRIC candidates only —
never [G] generative (you can't ground an ip2p_instruction without seeing the
image). Target [1] priority metric. Avoid near-zero strengths and anything
HARD-BLOCKED. Set "primary_index" to your best guess; the others are
safety-net hypotheses that get tested empirically regardless.
Respond ONLY with JSON: {{"primary_index":0,"candidates":[{{"category":"<tool>","strength":<float>,"why":"<≤12 words>","targeting":"<metric>","intent":"<word>"}}]}}"""
    return _sanitize_candidates(_groq_json(prompt, temperature=TEMP_CRITIC, max_tokens=700))

def _parse_json_response(text: str) -> dict:
    text=text.strip()
    if "```" in text:
        for part in text.split("```"):
            part=part.strip().lstrip("json").strip()
            if part.startswith("{"): text=part; break
    s=text.find("{"); e=text.rfind("}")+1
    if s!=-1 and e>s: text=text[s:e]
    return json.loads(text)

def _sanitize_candidates(dec: dict) -> dict:
    """Normalise an LLM response into {"primary_index":int,"candidates":[...]}.
    Accepts the multi-candidate schema or a legacy single-object response."""
    cands = dec.get("candidates")
    if not isinstance(cands, list) or not cands:
        single = {k: dec.get(k) for k in
                  ("category","strength","why","artistic_rationale","targeting","intent","ip2p_instruction")}
        cands = [single]
    clean=[]
    for c in cands:
        if not isinstance(c, dict): continue
        cat = c.get("category")
        if cat not in EDIT_CATEGORIES: continue
        try: stre = max(-1.0, min(1.0, float(c.get("strength", 0.0))))
        except (TypeError, ValueError): stre = 0.0
        cc = dict(c); cc["category"]=cat; cc["strength"]=stre
        # v8 schema uses "why"; keep "artistic_rationale" populated for downstream
        if not cc.get("artistic_rationale"): cc["artistic_rationale"]=cc.get("why","")
        clean.append(cc)
    if not clean:
        raise ValueError(f"No valid candidates in LLM response: {dec}")
    clean = clean[:MAX_PARAMETRIC_CANDIDATES]
    pidx = dec.get("primary_index", 0)
    if not isinstance(pidx, int) or not (0 <= pidx < len(clean)): pidx = 0
    dec["candidates"]=clean; dec["primary_index"]=pidx
    return dec

def _parse_redirect(note: str) -> dict | None:
    if not note.upper().startswith("REDIRECT"): return None
    nl=note.lower().replace("-","_")
    for cat in EDIT_CATEGORIES:
        if cat in nl:
            strs=re.findall(r"[+-]?\d+\.?\d*",note)
            valid=[float(x) for x in strs if -1.0<=float(x)<=1.0]
            return {"primary_index":0,"candidates":[{
                "category":cat,"strength":valid[0] if valid else 0.3,
                "artistic_rationale":f"Critic redirect: {note[:120]}",
                "targeting":"composite","intent":"redirect"}]}
    return None

def _deliberate(img, scores, history, attempt_log, scene_desc, blocked: set):
    # R1: Gemini proposes (model fallback + key rotation handled inside the call)
    proposal=None
    try:
        proposal=_gemini_propose(img,scores,history,attempt_log,scene_desc,blocked=blocked)
        prim=proposal["candidates"][proposal["primary_index"]]
        if proposal.get("see"): print(f"    👁  artist sees: {str(proposal['see'])[:150]}")
        print(f"    R1 Gemini[{(LAST_MODELS['gemini'] or '?').split('/')[-1]}] → "
              f"{len(proposal['candidates'])} cand., primary="
              f"{prim['category']:<22} {prim['strength']:+.2f}  [{prim.get('intent','?')}]")
    except Exception as e:
        print(f"    R1 Gemini failed (all models): {e}")

    if proposal is None:
        try:
            dec=_groq_decide_alone(scores,history,attempt_log,scene_desc,blocked=blocked)
            dec["_path"]="groq-only"; dec["_model"]=LAST_MODELS["groq"]; return dec
        except Exception as e:
            raise RuntimeError(f"Both LLMs unavailable: {e}")

    proposal["_model"]=LAST_MODELS["gemini"]
    # The blind critic only reviews where it can add value. For PARAMETRIC steps
    # every candidate is executed and scored anyway — that IS the critique.
    primary = proposal["candidates"][proposal["primary_index"]]
    need_critic = (CRITIC_MODE == "always" or
                   (CRITIC_MODE == "generative-only"
                    and EDIT_CATEGORIES.get(primary["category"]) == "GENERATIVE"))
    if not need_critic:
        proposal["_path"]="gemini-solo"; return proposal

    # R2: Groq critiques the primary candidate
    critic=None
    try:
        critic=_groq_critique(proposal,scores,history,attempt_log,scene_desc)
        verdict=critic.split()[0].upper() if critic else "?"
        print(f"    R2 Groq[{(LAST_MODELS['groq'] or '?').split('/')[-1]}] → {verdict:<8}  {critic[:70]}…")
    except Exception as e:
        print(f"    R2 Groq failed: {e} — using R1")
        proposal["_path"]="gemini-only"; return proposal

    if critic.upper().startswith("AGREE"):
        proposal["critic_note"]=critic; proposal["_path"]="gemini+groq-agree"
        return proposal

    redirect=_parse_redirect(critic)
    if redirect:
        print(f"    R2 redirect parsed → {redirect['candidates'][0]['category']} "
              f"{redirect['candidates'][0]['strength']:+.2f}")

    # R3: Gemini reconsiders with the critique (only on MODIFY/REDIRECT)
    try:
        final=_gemini_propose(img,scores,history,attempt_log,scene_desc,critic,blocked=blocked)
        final["critic_note"]=critic; final["_path"]="full-deliberation"; final["_model"]=LAST_MODELS["gemini"]
        prim=final["candidates"][final["primary_index"]]
        print(f"    R3 Gemini → {len(final['candidates'])} cand., primary="
              f"{prim['category']:<22} {prim['strength']:+.2f}  [{prim.get('intent','?')}]")
        return final
    except Exception as e:
        print(f"    R3 Gemini failed: {e}")
        if redirect:
            redirect["critic_note"]=critic; redirect["_path"]="critic-redirect"; return redirect
        proposal["critic_note"]=critic; proposal["_path"]="gemini+groq-r1"; return proposal

# --- Vision reality-check: pairwise A/B with both orderings --------------------
def _gemini_ab(img1: Image.Image, img2: Image.Image) -> int:
    prompt=("You are judging an image edit. Image 1 and Image 2 are two versions of the "
            "same picture. Which is aesthetically BETTER — more pleasing tonality, colour, "
            "depth, mood and craft? Reward bold, intentional, beautiful looks. Penalise "
            "oversharpening, oversaturation, halos, crushed or washed-out tones and an "
            'artificial HDR look. Respond ONLY with JSON: {"winner": 1 or 2, "reason": "<one sentence>"}')
    txt=_gemini_text([{"text":prompt},_img_part(img1),_img_part(img2)],
                     temperature=TEMP_JUDGE, max_tokens=2048)
    return int(_loads_lenient(txt).get("winner", 0))

def _vision_prefers_candidate(base_img: Image.Image, cand_img: Image.Image):
    """True = judge prefers cand in BOTH orderings; False = base in both;
    None = split decision or judge unavailable."""
    try:
        w1=_gemini_ab(base_img,cand_img); w2=_gemini_ab(cand_img,base_img)
    except Exception as e:
        print(f"    [vision-check] unavailable: {e}")
        return None
    votes_cand=(1 if w1==2 else 0)+(1 if w2==1 else 0)
    if votes_cand==2: return True
    if votes_cand==0: return False
    return None

# %% [code] {"jupyter":{"outputs_hidden":false}}
def _is_acceptable(scores: dict, prev_scores: dict, edit_type: str, step: int) -> tuple[bool,str]:
    if edit_type=="GENERATIVE":
        # v8: gate on CLIP SEMANTIC drift (same scene/subject?), NOT SSIM —
        # so recolour/relight/restyle is free; only a different picture is blocked.
        sem=scores.get("sem_drift");  cum_sem=scores.get("cum_sem_drift")
        if sem is not None and sem>SEM_DRIFT_HARD:
            return False,f"semantic drift {sem:.3f} > {SEM_DRIFT_HARD} (became a different image)"
        if cum_sem is not None and cum_sem>CUM_SEM_DRIFT_CAP:
            return False,f"cumulative semantic drift {cum_sem:.3f} > {CUM_SEM_DRIFT_CAP}"
        if sem is not None and sem>SEM_DRIFT_SOFT:
            gain=(scores["composite"] or 0)-(prev_scores["composite"] or 0)
            if gain<SEM_GAIN_MIN:
                return False,f"semantic soft-zone ({sem:.3f}); gain {gain:.4f} < {SEM_GAIN_MIN}"
        # loose SSIM backstop only — catches an output that's pure noise/garbage.
        if scores.get("struct_drift",0.0)>STRUCT_DRIFT_BACKSTOP:
            return False,f"SSIM backstop {scores['struct_drift']:.3f} > {STRUCT_DRIFT_BACKSTOP}"
    # Generative experiments get a more generous floor than parametric tweaks.
    floor=GEN_EXPLORATION_FLOOR if edit_type=="GENERATIVE" else _exploration_floor(step)
    delta=(scores["composite"] or 0)-(prev_scores["composite"] or 0)
    if delta<floor:
        return False,f"composite dropped {delta:.4f} (floor {floor:+.4f})"
    # Held-out CLIP-IQA: ADVISORY by default in v9 (it false-flagged the artist's
    # correct warm-up edits in the v8 run, and it's photo-trained too). Only a
    # hard gate if HELDOUT_AS_GATE — otherwise PickScore is the anti-gaming guard.
    if HELDOUT_AS_GATE:
        hc=scores.get("heldout"); hp=prev_scores.get("heldout")
        htol=HELDOUT_TOLERANCE_GEN if edit_type=="GENERATIVE" else HELDOUT_TOLERANCE
        if hc is not None and hp is not None and hc < hp - htol:
            return False,f"held-out validator dropped {hc-hp:+.4f} (tol -{htol}) — likely metric gaming"
    return True,"ok"

def _expand_parametric(cands: list, primary_idx: int) -> list:
    """Strength sweep: each LLM candidate also gets ×0.65 / ×1.45 variants.
    LLM-proposed strengths first (primary first), then variants, deduped,
    capped at MAX_PARAMETRIC_EVALS. Scoring is cheap; LLM calls are not —
    one call now buys a small line search instead of a single guess."""
    def _cap(cat, s):  # v10: clamp destructive tools to their per-category max
        m=MAX_STRENGTH_BY_CATEGORY.get(cat, 1.0)
        return max(-m, min(m, s))
    out=[]; seen=set()
    def add(c, is_primary, tag):
        s=_cap(c["category"], c["strength"])
        key=(c["category"], round(s,2))
        if key in seen or abs(s)<0.02: return
        seen.add(key)
        cc=dict(c); cc["strength"]=s; cc["_sweep"]=tag; cc["_is_primary"]=is_primary
        out.append(cc)
    order=[primary_idx]+[i for i in range(len(cands)) if i!=primary_idx]
    for i in order:
        add(cands[i], i==primary_idx, "llm")
    for m in STRENGTH_SWEEP:
        if abs(m-1.0)<1e-6: continue
        for i in order:
            c=cands[i]
            # don't let the up-sweep push destructive tools harder
            if m>1.0 and c["category"] in NO_UPSWEEP_CATEGORIES: continue
            s=max(-1.0,min(1.0,round(c["strength"]*m,2)))
            add({**c,"strength":s}, False, f"sweep×{m:g}")
    return out[:MAX_PARAMETRIC_EVALS]

def _make_contact_sheet(frames: list, cols: int, thumb: int) -> Image.Image:
    """Grid montage of the evolution frames, each captioned with its label and
    the key anime/overall scores, so the contribution of each stage is visible."""
    from math import ceil
    cap=52; n=len(frames); rows=ceil(n/cols)
    cw,ch=thumb,thumb+cap
    sheet=Image.new("RGB",(cols*cw,rows*ch),(20,20,24)); draw=ImageDraw.Draw(sheet)
    for i,f in enumerate(frames):
        im=f["img"].copy(); im.thumbnail((thumb,thumb))
        r,c=divmod(i,cols); x=c*cw; y=r*ch
        sheet.paste(im,(x+(cw-im.width)//2,y))
        sc=f["scores"]; nm=sc.get("normed",{})
        draw.text((x+5,y+thumb+3), f["label"], fill=(235,235,240))
        line2=f"comp {sc.get('composite',0):.3f}"
        if nm.get("anime_aesthetic") is not None: line2+=f"  anime {nm['anime_aesthetic']:.2f}"
        if nm.get("shadow_aesthetic") is not None: line2+=f"  illu {nm['shadow_aesthetic']:.2f}"
        draw.text((x+5,y+thumb+22), line2, fill=(150,160,175))
    return sheet

def run_pipeline(source: Image.Image) -> dict:
    current_img=source.copy(); best_img=source.copy()
    best_scores=dict(SOURCE_SCORES); prev_scores=dict(SOURCE_SCORES)
    history=[]; attempt_log=[]; no_improve=0.0; restarts=0; step=0
    last_accepted=source.copy(); any_accepted=False
    # Evolution: the actual walked trajectory (source + every accepted edit).
    evolution=[{"step":-1,"label":"source","img":source.copy(),"scores":dict(SOURCE_SCORES)}]
    # Judge state: last image the human-preference judge approved (checkpoint).
    judge = "pickscore" if _pick_model is not None else ("gemini" if VISION_CHECK_EVERY>0 else "off")
    ckpt_img=source.copy(); ckpt_scores=dict(SOURCE_SCORES)
    ckpt_pick=_pickscore(source) if judge=="pickscore" else None
    accepted_since_check=0; vision_note=""; early_stop=False

    print("\nGenerating scene description …")
    try:
        scene=_gemini_describe_scene(source)
        print(f"  ↳ {scene[:200]}…\n")
    except Exception as e:
        print(f"  [warn] {e}"); scene="An image requiring aesthetic improvement."
    orig_scene=scene

    print(f"{'═'*72}")
    print(f"  DUAL ARTIST v10  ·  {MAX_EDIT_STEPS} steps  ·  patience {RESTART_PATIENCE}  ·  {MAX_RESTARTS} restarts")
    print(f"  Source composite: {SOURCE_SCORES['composite']:.4f}   held-out: {SOURCE_SCORES.get('heldout')}")
    print(f"  Editor: {_gen_backend or 'none'}  ·  Judge: {judge}  ·  critic: {CRITIC_MODE}  ·  creative: {CREATIVE_MODE}")
    print(f"  Floor: P {EXPLORATION_FLOOR_START:+.3f}→{EXPLORATION_FLOOR_END:+.3f} / G {GEN_EXPLORATION_FLOOR:+.3f}")
    print(f"  Evals/step: ≤{MAX_PARAMETRIC_EVALS} (LLM cands × strength sweep)  ·  Generative: single-shot")
    print(f"  Generative gate: SEMANTIC drift {SEM_DRIFT_HARD}/{SEM_DRIFT_SOFT}, cum {CUM_SEM_DRIFT_CAP}  ·  colour/mood FREE")
    print(f"  Anti-gaming: metric cap {MAX_METRIC_WEIGHT}  ·  prefer-artist-primary (margin {PREFER_PRIMARY_MARGIN})  ·  vignette≤{MAX_STRENGTH_BY_CATEGORY.get('vignette')}")
    print(f"{'═'*72}\n")

    while step<MAX_EDIT_STEPS:
        # v6: no loop-top rescore — prev_scores IS the current image's scores.
        # v7: no scene refresh — the artist reads the attached current image itself.
        scores=prev_scores

        merged_profile=_merge_profiles(PRIOR_TOOL_PROFILE,_tool_profile_dict(attempt_log))
        blocked=_blocked_set(merged_profile)

        print(f"  step {step:02d}  deliberating …")
        try:
            dec=_deliberate(current_img,scores,history,attempt_log,scene+vision_note,blocked)
        except Exception as e:
            print(f"  step {step:02d}  [error] {e}"); step+=1; no_improve+=1; continue

        candidates=dec.get("candidates") or []
        if not candidates:
            print(f"  step {step:02d}  [skip] no usable candidates"); step+=1; no_improve+=1; continue
        primary_idx=dec.get("primary_index",0)
        if not (0<=primary_idx<len(candidates)): primary_idx=0
        primary=candidates[primary_idx]
        primary_cat=primary["category"]; primary_etype=EDIT_CATEGORIES[primary_cat]
        critic_note=dec.get("critic_note",""); path=dec.get("_path","?")
        floor=_exploration_floor(step)
        prev_snapshot=dict(prev_scores)

        results=[]
        if primary_etype=="GENERATIVE":
            stre=primary["strength"]
            instruction=primary.get("ip2p_instruction") or _fallback_ip2p_instruction(primary_cat,stre)
            try:
                cand_img=run_generative_edit(current_img,instruction,stre)
            except Exception as e:
                print(f"  step {step:02d}  [error] generative {primary_cat}: {e}"); step+=1; no_improve+=1; continue
            cs=_compute_scores(cand_img, drift_ref=current_img, semantic=True)
            accept,reason=_is_acceptable(cs,prev_scores,"GENERATIVE",step)
            results.append({"candidate":primary,"is_primary":True,"img":cand_img,
                             "scores":cs,"accepted":accept,"reason":reason,"instruction":instruction})
        else:
            pcands=[c for c in candidates if EDIT_CATEGORIES.get(c["category"])=="PARAMETRIC"]
            if not pcands: pcands=[primary]
            pidx=pcands.index(primary) if primary in pcands else 0
            expanded=_expand_parametric(pcands,pidx)
            kept=[c for c in expanded if not _is_blocked(c,blocked)]
            if kept and len(kept)<len(expanded):
                print(f"           ⛔ {len(expanded)-len(kept)} candidate(s) filtered by category block")
            if not kept: kept=expanded   # never deadlock on blocking
            for c in kept:
                cat=c["category"]; stre=c["strength"]
                try:
                    cand_img=apply_edit(current_img,cat,stre)
                except Exception as e:
                    print(f"  step {step:02d}  [error] apply {cat}: {e}"); continue
                cs=_compute_scores(cand_img, drift_ref=current_img)
                accept,reason=_is_acceptable(cs,prev_scores,"PARAMETRIC",step)
                results.append({"candidate":c,"is_primary":c.get("_is_primary",False),"img":cand_img,
                                "scores":cs,"accepted":accept,"reason":reason})
            if not results:
                print(f"  step {step:02d}  [skip] all candidates failed to apply"); step+=1; no_improve+=1; continue

        # Log EVERY attempt (feeds tool-effect profile, calibration, blocking)
        for i,r in enumerate(results):
            c=r["candidate"]
            attempt_log.append({
                "step":step,"candidate_idx":i,"is_primary":r["is_primary"],
                "sweep":c.get("_sweep","llm"),
                "category":c["category"],"edit_type":EDIT_CATEGORIES[c["category"]],
                "strength":c["strength"],"rationale":c.get("artistic_rationale","") or c.get("why",""),
                "targeting":c.get("targeting","—"),"intent":c.get("intent","—"),
                "critic_note":critic_note,"deliberation":path,"model":dec.get("_model"),
                "scores":r["scores"],"prev_scores":prev_snapshot,
                "accepted":r["accepted"],"reason":r["reason"],
            })

        acceptable=[r for r in results if r["accepted"]]
        # v10: prefer the ARTIST'S primary choice — only let a sibling win if it
        # beats the primary by more than PREFER_PRIMARY_MARGIN. This stops pure
        # metric-maximisation from hijacking the artist's intent (vignette-spam
        # in v9). If the primary wasn't acceptable, fall back to best composite.
        chosen=None
        if acceptable:
            best_r=max(acceptable,key=lambda r:r["scores"]["composite"])
            prim_r=next((r for r in acceptable if r.get("is_primary")), None)
            if prim_r is not None and (best_r["scores"]["composite"]
                                       - prim_r["scores"]["composite"] <= PREFER_PRIMARY_MARGIN):
                chosen=prim_r
            else:
                chosen=best_r
        rep = chosen or max(results,key=lambda r:r["scores"]["composite"])

        for r in results:
            c=r["candidate"]; cs=r["scores"]
            tag="★" if chosen is r else ("·" if r["accepted"] else "✗")
            mark="primary " if r["is_primary"] else (f"{c.get('_sweep','sib'):<8}" if c.get("_sweep","llm")!="llm" else "sibling ")
            dp=cs["composite"]-prev_scores["composite"]
            dft=(f"sem {cs['sem_drift']:.3f}" if cs.get("sem_drift") is not None
                 else f"ssim {cs['struct_drift']:.3f}")
            print(f"           {tag} {mark}{c['category']:<22} {c['strength']:+.2f}  "
                  f"composite {cs['composite']:.4f} (Δprev {dp:+.4f})  {dft}  "
                  f"[{c.get('intent','—')}]")
            if not r["accepted"]: print(f"               ✗ {r['reason']}")
        if primary_etype=="GENERATIVE":
            print(f"           ↳ {_gen_backend or 'gen'}: \"{results[0]['instruction']}\"")
        _why=primary.get("why") or primary.get("artistic_rationale")
        if _why: print(f"           ↳ {_why[:140]}")
        print(f"           ({path}, {len(results)} candidate(s), floor {floor:+.3f})")

        history.append({"step":step,"category":rep["candidate"]["category"],"strength":rep["candidate"]["strength"],
                         "intent":rep["candidate"].get("intent","—"),"accepted":bool(chosen),
                         "reason":rep["reason"],"scores":rep["scores"],"n_candidates":len(results),
                         "model":dec.get("_model")})

        ds_=rep["scores"]["composite"]-SOURCE_SCORES["composite"]
        if chosen:
            current_img=chosen["img"].copy(); prev_scores=dict(chosen["scores"])
            last_accepted=chosen["img"].copy(); any_accepted=True
            no_improve=max(0.0,no_improve-0.5)
            _is_best = chosen["scores"]["composite"]>best_scores["composite"]+1e-6
            # Capture this accepted stage for the evolution montage.
            evolution.append({"step":step,
                "label":f"s{step:02d} {rep['candidate']['category'][:12]}{' ★' if _is_best else ''}",
                "img":chosen["img"].copy(),"scores":dict(chosen["scores"])})
            if _is_best:
                best_img=chosen["img"].copy(); best_scores=dict(chosen["scores"]); no_improve=0
                print(f"           ★ NEW BEST {best_scores['composite']:.4f}  (Δsrc {ds_:+.4f})")
                if EARLY_STOP_ON_TARGETS and _targets_met(best_scores):
                    print(f"\n  ✓ All weighted metrics at target — stopping early.")
                    early_stop=True
            else:
                no_improve+=1

            # --- The judge (anti reward-hacking) ---------------------------
            _rolled_back=False
            if judge=="pickscore" and not early_stop:
                # Local human-preference judge: gate EVERY accepted edit.
                pick_new=_pickscore(current_img)
                p=_pick_prefers_new(pick_new, ckpt_pick)
                if p is not None:
                    if p < 0.5 - PICK_MARGIN:
                        print(f"           ⚖ PickScore REJECT (p_new={p:.2f}, "
                              f"{pick_new:.3f} vs ckpt {ckpt_pick:.3f}) — metrics rose, "
                              f"preference fell. Rolling back.")
                        current_img=ckpt_img.copy(); prev_scores=dict(ckpt_scores)
                        best_img=ckpt_img.copy();    best_scores=dict(ckpt_scores)
                        last_accepted=ckpt_img.copy(); no_improve=0; _rolled_back=True
                        vision_note=("\n\n⚠ JUDGE ROLLBACK: your last edit raised the metric scores but a "
                                     "human-preference judge (PickScore) liked it LESS than the previous state "
                                     "— classic metric gaming (oversharpening / oversaturation / artificial HDR). "
                                     "It was UNDONE. Use a subtler strength or a different category; prefer "
                                     "edits that look better to the eye, not just to the scorers.")
                    else:
                        ckpt_img=current_img.copy(); ckpt_scores=dict(prev_scores)
                        ckpt_pick=pick_new; vision_note=""
                        print(f"           ⚖ PickScore OK (p_new={p:.2f}) — checkpoint advanced")
            elif judge=="gemini" and not early_stop:
                # Fallback: periodic Gemini A/B (v6 behaviour) when PickScore is down.
                accepted_since_check+=1
                if accepted_since_check>=VISION_CHECK_EVERY:
                    accepted_since_check=0
                    print(f"           ⚖ Gemini A/B check vs last approved state …")
                    v=_vision_prefers_candidate(ckpt_img,current_img)
                    if v is True:
                        ckpt_img=current_img.copy(); ckpt_scores=dict(prev_scores); vision_note=""
                        print(f"           ⚖ confirmed — checkpoint advanced")
                    elif v is False:
                        print(f"           ⚖ REJECTED — metrics rose but the image looks WORSE. Rolling back.")
                        current_img=ckpt_img.copy(); prev_scores=dict(ckpt_scores)
                        best_img=ckpt_img.copy();    best_scores=dict(ckpt_scores)
                        last_accepted=ckpt_img.copy(); no_improve=0; _rolled_back=True
                        vision_note=("\n\n⚠ VISION REALITY-CHECK: your recent accepted streak raised the metrics "
                                     "but was judged WORSE by direct visual comparison (metric gaming). The state "
                                     "was ROLLED BACK. Use subtler strengths and different categories.")
                    else:
                        print(f"           ⚖ split decision — state kept, checkpoint unchanged")

            # Fast restart (config existed in v5, implemented in v6): accepted
            # but far below best → snap back now instead of bleeding patience.
            if (not early_stop and not _rolled_back and
                best_scores["composite"]-prev_scores["composite"]>FAST_RESTART_THRESHOLD):
                if restarts<MAX_RESTARTS:
                    restarts+=1; no_improve=0
                    current_img=best_img.copy(); prev_scores=dict(best_scores)
                    print(f"\n  [fast-restart {restarts}/{MAX_RESTARTS}] accepted edit fell "
                          f">{FAST_RESTART_THRESHOLD} below best → back to {best_scores['composite']:.4f}\n")
        else:
            no_improve+=0.5

        if early_stop: break

        if no_improve>=RESTART_PATIENCE:
            if restarts>=MAX_RESTARTS: print(f"\n  Max restarts reached — stopping."); break
            restarts+=1; no_improve=0
            current_img=best_img.copy(); prev_scores=dict(best_scores)
            print(f"\n  [restart {restarts}/{MAX_RESTARTS}] → best composite: {best_scores['composite']:.4f}\n")
        step+=1

    # Final vision verdict: does the visual judge actually prefer the result?
    final_verdict="not-run"
    if VISION_CHECK_EVERY>0 and best_scores["composite"]>SOURCE_SCORES["composite"]+1e-4:
        print("\n  ⚖ FINAL vision A/B: source vs best …")
        v=_vision_prefers_candidate(source,best_img)
        final_verdict={True:"best-confirmed",False:"source-preferred",None:"split"}.get(v,"split")
        print(f"  ⚖ verdict: {final_verdict}")
        if v is False:
            print("  ⚠ The visual judge prefers the ORIGINAL despite higher metrics — inspect outputs by eye.")

    delta=best_scores["composite"]-SOURCE_SCORES["composite"]
    n_gen=sum(1 for h in history if EDIT_CATEGORIES.get(h.get("category",""))=="GENERATIVE")
    n_gen_ok=sum(1 for h in history if h.get("accepted") and EDIT_CATEGORIES.get(h.get("category",""))=="GENERATIVE")
    print(f"\n{'═'*72}")
    print(f"  DONE  ·  {step} steps  ·  {restarts} restarts  ·  {len(attempt_log)} candidate attempts")
    print(f"  Source:  {SOURCE_SCORES['composite']:.4f}  →  Best: {best_scores['composite']:.4f}  ({delta:+.4f})")
    print(f"  Held-out: {SOURCE_SCORES.get('heldout')} → {best_scores.get('heldout')}   ·  vision verdict: {final_verdict}")
    print(f"  Generative edits: {n_gen_ok}/{n_gen} accepted  ·  final cum. semantic drift: {best_scores.get('cum_sem_drift')}")
    print(f"  Models used — Gemini: {gemini_roster.report()}")
    print(f"              — Groq:   {groq_roster.report()}")
    print(f"{'═'*72}")
    print(f"  Evolution: {len(evolution)} stages captured (source + {len(evolution)-1} accepted edits)")
    return {"best_image":best_img,"last_accepted_image":last_accepted,"any_accepted":any_accepted,
            "source_scores":SOURCE_SCORES,"best_scores":best_scores,
            "log":attempt_log,"history":history,"evolution":evolution,
            "steps":step,"restarts":restarts,"scene_description":orig_scene,
            "vision_verdict":final_verdict,"early_stop":early_stop,
            "models":{"gemini":gemini_roster.report(),"groq":groq_roster.report()},
            "tool_profile":_tool_profile_dict(attempt_log)}

# %% [code] {"jupyter":{"outputs_hidden":false}}
result=run_pipeline(source_image)

# PNG: the pipeline just spent the whole run optimising aesthetics — don't
# hand part of it back to JPEG re-encoding (v5 saved at quality 95).
result["best_image"].save("/kaggle/working/best_output.png")
print("✓ best_output.png saved")
if result["any_accepted"]:
    result["last_accepted_image"].save("/kaggle/working/last_accepted_output.png")
    print("✓ last_accepted_output.png saved")
else:
    print("  ⚠ No edits accepted.")

with open("/kaggle/working/edit_log.json","w") as f:
    json.dump(result["log"],f,indent=2,default=str)
print(f"✓ edit_log.json saved ({len(result['log'])} candidate attempts across {result['steps']} steps)")

with open("/kaggle/working/scene_description.txt","w") as f: f.write(result["scene_description"])
print("✓ scene_description.txt saved")

# Evolution: individual stage PNGs + one labelled contact sheet so you can SEE
# whether each accepted stage actually contributed.
if SAVE_EVOLUTION and result.get("evolution"):
    import os as _os
    _os.makedirs(EVOLUTION_DIR, exist_ok=True)
    for _i,_f in enumerate(result["evolution"]):
        _f["img"].save(f"{EVOLUTION_DIR}/{_i:02d}_{_f['label'].replace(' ','_').replace('★','best')}.png")
    try:
        _sheet=_make_contact_sheet(result["evolution"], EVOLUTION_COLS, EVOLUTION_THUMB)
        _sheet.save("/kaggle/working/evolution_contactsheet.png")
        print(f"✓ evolution: {len(result['evolution'])} stage PNGs in {EVOLUTION_DIR}/ + evolution_contactsheet.png")
    except Exception as _e:
        print(f"  [warn] contact sheet failed ({_e}); individual frames still saved")

# Persist the tool-effect profile, merged with any PRIOR_TOOL_PROFILE — carry
# tool_profile.json forward as PRIOR_TOOL_PROFILE_PATH next run to keep
# compounding "understanding of the editor" across different images.
_merged_profile=_merge_profiles(PRIOR_TOOL_PROFILE, result["tool_profile"])
with open(TOOL_PROFILE_PATH,"w") as f: json.dump(_merged_profile,f,indent=2)
_n_series=sum(len(m) for m in _merged_profile.values())
print(f"✓ tool_profile.json saved ({len(_merged_profile)} tool-directions, {_n_series} metric-series)")
print(f"  → set PRIOR_TOOL_PROFILE_PATH to this file's path next run to keep compounding it")
print(f"  → final vision verdict: {result['vision_verdict']}")
