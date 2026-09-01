# Context Length: Why Model H Uses 512 and Model L Uses 1024

Model H (Hindi) and Model L (Sindhi) are identical in every architectural respect except
context length. This note records why, what it costs, and how the language-modelling
comparison between the two is kept valid despite the difference.

## The measurement

Document lengths were measured on each language's held-out validation split, tokenized
with that language's own 16K Unigram tokenizer (20,000 documents per language):

| | median | mean | 90th pct | fits in 512 | fits in 1024 |
|---|---|---|---|---|---|
| **Hindi** | 64 tokens | 85 | 149 | **99.3%** | 99.8% |
| **Sindhi** | 219 tokens | 774 | 1,380 | **74.5%** | 86.4% |

The two corpora are shaped very differently. This is not a quirk of tokenization — the
fertility of the two tokenizers is nearly identical (1.3056 vs 1.2987 tokens/word), so the
gap reflects genuinely longer documents, not more tokens per word.

The reason is the sources. The Hindi corpus is dominated by news and encyclopaedia
paragraphs from IndicCorp and Sangraha, which are short by construction. The Sindhi corpus
draws heavily on long-form books, literary prose and full newspaper articles, because
those were the sources that existed in quantity for a lower-resource language.

## The decision

* **Model H: 512 tokens.** At a median of 64 tokens, a 512-token window already spans
  roughly eight Hindi documents. Extending it would mostly add text from unrelated
  neighbouring articles, which carries no useful predictive signal for the current
  document — after a `<|endoftext|>` boundary the earlier content is a different topic
  entirely. 99.3% of documents already fit whole.

* **Model L: 1024 tokens.** At a median of 219 and a mean of 774, a 512-token window
  frequently sits *inside* a single continuous Sindhi document, with the earlier half of
  the same passage outside the window. Here the additional context is real, same-topic
  history: the antecedent of a pronoun, a character introduced earlier in a chapter, the
  premise of an argument being continued. Raising the window to 1024 lifts whole-document
  coverage from 74.5% to 86.4%.

Choosing one number for both would mean either giving Model L a window that truncates its
documents, or giving Model H a window it demonstrably cannot use.

## What it costs

**Parameters.** The models differ by exactly 262,144 parameters, which is the extra
position table (512 additional rows x 512 dimensions) and nothing else:

| | Model H | Model L |
|---|---|---|
| Total parameters | 24,220,672 | 24,482,816 |
| Non-embedding parameters | 15,765,504 | 15,765,504 |
| d_model / layers / heads / d_ff | 512 / 6 / 8 / 1536 | 512 / 6 / 8 / 1536 |

The **non-embedding** parameter count — the transformer stack that does the actual
modelling — is byte-identical between the two models. `make_configs.py --check` asserts
this, and fails if the two configs ever differ by anything other than the position table.
Both models sit within the ~25M target.

**Compute.** Attention cost grows with the square of context length, so Model L is slower
per token. Measured on the training hardware (RTX 3050, bf16):

| | throughput | est. time for 500M tokens |
|---|---|---|
| Model H, T=512, batch 8 | 20,707 tok/s | ~6.7 h |
| Model L, T=1024, batch 4 | 14,678 tok/s | ~9.5 h |

The batch size drops from 8 to 4 to stay inside 4 GB of VRAM; gradient accumulation keeps
the *effective* batch identical between the two runs, so this costs wall-clock time only,
not optimisation behaviour.

## The threat to validity, and how it is handled

Context length is a confound for perplexity. A model with more history to condition on
will, all else equal, report lower loss — simply because it has seen more of the passage.
Comparing Model H at 512 against Model L at 1024 therefore does not, on its own, isolate
the resource-tier effect the project is asking about. Model L would receive an unearned
advantage.

Three things address this.

**1. A matched-context evaluation is reported alongside the native one.** Both models are
additionally evaluated scoring only positions with at most 511 tokens of preceding
context, which is the longest history Model H can ever have. At that setting the two
models are conditioned on identical amounts of history and the comparison is
apples-to-apples. The native-context numbers are reported too, since they describe how
each model actually performs in deployment, but the *resource-tier claim* rests on the
matched-context figures.

**2. Bits-per-byte is the primary cross-language metric.** Perplexity per token is not
comparable across models with different tokenizers in any case, which is why the brief
asks for BPB. BPB normalises by the underlying UTF-8 bytes rather than by tokens, so it is
already the right instrument for comparing two models whose vocabularies were learned
independently. It is reported under both native and matched context.

**3. The difference is stated, not hidden.** The corpus-shape asymmetry is itself a
finding, and one the Phase 3 report is explicitly asked to explain: "What tokenizer /
corpus factors most affected the lower-resource model?" That Sindhi's available public
text is long-form literary and news prose while Hindi's is short paragraphs is a real,
measurable consequence of the resource gap, and it shapes what each model can learn.

## Alternative considered

Using 512 for both would have made the comparison maximally clean at the cost of
handicapping Model L on the corpus it actually has. Using 1024 for both would have cost an
extra ~2.8 hours of training on Model H to buy context it cannot use, and would have made
the Hindi model's position table 262K parameters larger for no benefit.

The chosen configuration accepts a controllable confound — controllable because the
matched-context evaluation removes it — in exchange for each model being sized to its own
data.
