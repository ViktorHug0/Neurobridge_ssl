---
name: paper-prose
description: Rules for writing or editing academic paper text (LaTeX papers, rebuttals, abstracts, section titles, captions) so it reads as human-written rather than LLM-generated. Use whenever drafting, rewriting or reviewing prose in CVPR_paper/, Neurips_paper_and_rebuttal/ or any other paper in this repo.
---

# Writing paper prose that does not read as LLM-generated

The authors reject text that sounds machine-written, even when its content is correct. Apply these rules to every heading, caption and sentence you write, then reread your text against the checklist at the end.

## Section and paragraph titles

A heading names the object of study as a plain noun phrase. It never states a result.

| Wrong (LLM register) | Why it reads as artificial | Right |
|---|---|---|
| A setting and a protocol. | paired slogan, no content | (no label) / Problem setting |
| What the gain is made of. | explainer-style "what X" clause, idiom | Analysis |
| What the Calibration Exploits | same, plus anthropomorphic verb | Analysis |
| How Many Labels Is Unlabeled Calibration Worth? | question and wordplay | Comparison with Supervised Alignment |
| What is, and is not, an additional assumption. | rhetorical contrast | Assumptions |
| The displacement of a new subject. | vague metaphorical noun | Form of the cross-subject shift |
| X registers the test set; it does not recover the displacement. | full claim, semicolon, "X, not Y" | Transfer to unseen images |
| RSA is not a training objective. | claim as heading, negation | RSA as a training loss |
| Beyond the bijective test (figure caption) | "Beyond X" cliché | Streaming and open-gallery results |
| Which encoders benefit from calibration. (caption) | question-like teaser | Calibration gain across encoders |

Contribution lists: plain items starting with "We ...". No bold catchphrase labels.

## Sentences

Avoid these patterns. Each one is a strong LLM marker.

- **Antithesis endings**: "X, not Y", "rather than Y", "not A but B", "X; it does not Y".
  - Wrong: "what it contributes is the alignment of the geometry, not the assignment."
  - Right: "SAGE is unaffected by the Hungarian decoding and remains 18 points above the prior."
- **Aphoristic morals that close a paragraph.**
  - Wrong: "The calibration amplifies the signal present in the embeddings and does not compensate for its absence."
  - Right: "SAGE therefore requires queries with a sufficient signal-to-noise ratio."
- **Defensive meta-statements.**
  - Wrong: "We take no position on the origin of the gap. We ... treat its success as a property of the correction rather than as evidence about the cause."
  - Right: "In this work, we study a correction estimated from unlabeled data, which recovers part of the lost accuracy without making assumptions on the origin of the gap."
- **Punchy one-liners and metaphors**: "A deployed decoder sits between the two.", "offers a stronger handle than either", "which locates the problem in ...", "silently assumes".
- **Overclaiming triads**: "so their numbers are not comparable and their failure modes are unknown" becomes "which makes their results difficult to compare".
- **Vague abstract nouns** where a technical term exists: displacement (shift), handle, made of, exploits (uses / relies on).
- **Number dumps.** Give only the numbers needed to read the outcome of an experiment. One or two per claim, not five sizes in a row.

## Positive guidance

- Match the register of the authors' own text. Copy their NeurIPS sentences when they exist (`Neurips_paper_and_rebuttal/paper.tex`).
- Plain declarative sentences: subject, verb, result, with "thus" or "therefore" when a conclusion follows.
- One idea per sentence. Split run-ons that chain three findings with commas.
- Keep hedges factual ("on ATM", "over three seeds"), not rhetorical.

## Checklist before handing text back

1. Does any heading state a conclusion, ask a question, or start with "What"/"How"? Rename it to a noun phrase.
2. Search for `rather than`, `, not `, `; it `, `not .* but`, `Beyond`, `by itself`, `silently`. Rewrite each hit unless it is a plain technical contrast.
3. Does any paragraph end on a quotable moral? Replace it with the plain consequence, or delete it.
4. Is every number needed? Cut the rest.
