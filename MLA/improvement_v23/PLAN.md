> User time amendment: finish within one elapsed hour, conservatively counting from 22:23:09 IST (deadline 23:23:09). Engineering cutoff 22:53:09; training cutoff 23:13:09. This supersedes the initial plan before any V23 prediction or accuracy result. Initial documents are preserved with `initial_` prefixes. The two-member variant is a speed experiment, not the author benchmark configuration.

# V23: one controlled LimiX-2 regression experiment

Started October 29, 2026, IST, following the user's request to research deeply and start the best next improvement. This is a new experiment; V22 and its elapsed ledger remain unchanged. The one-hour ceiling is a conservative operational limit, including research, engineering, interruptions, evaluation and delivery. No paid resource or automatic portal upload.

V14 remains the observed best: RMSE 11,736,110.7877709. The latest available screenshot records rank 11 on October 29 at 19:44:46; this is not a current leaderboard query. V22's exploratory confirmation averaged only 0.017026% relative random-split gain, lost one random split, and had a paired interval spanning zero. Repeating that feature ablation is not the V23 hypothesis.

## Chosen hypothesis

An untested, independently pretrained LimiX-2 component may add useful errors different from TabPFN-3.5. This is a hypothesis, not a promised improvement. Use **L2-TF2**: the 406M-parameter LimiX-2 (not the older 2M-parameter LimiX-2M), two fixed official regression transformations (original members 0 and 4: quantile/SVD and power/one-hot), original V14 forty-column feature view, seed 10242, raw labels in INR millions, float32, no mixed precision, predictive regression output, and nonnegative outputs. Preserve all 3,493 labels.

Source revision: `c76fa21bbb25fddca649e75a52dc9b1df66a659c`. Checkpoint: `stable-ai/LimiX-2`, revision `de07b679e74a41b50b9de18251a8fa245e537440`, file `LimiX-2.ckpt`, SHA256 `ed01f7a0a18d25451bc9d1058b479ba6b68048aa1ebde06156c8fcfaf8b8fe81`.

The source audit found that regression's initial category conversion, category inference, feature filtering and categorical transformers can inspect concatenated support/query rows. **TF is an explicit train-only adapter**, not an assertion that this is the unchanged official default. Fit vocabularies, categorical inference, feature masks and categorical transformers on fitting rows; transform queries with fixed state. Keep the official distribution transforms, SVD and feature shuffles. No query labels, IDs, target caps, target deletion, extra features or accuracy-driven settings search. Log the adapter and source identities.

## Freeze before evaluation

Test only `candidate(w) = (1-w)*V14 + w*L2-TF2`, with one total weight **0.25**. All other settings are fixed. At most ten new development folds, one fixed blend, and one locked winner. No substitute family if engineering fails.

Before accuracy screening: verify the original input/reference hashes and all 35 matched original P35 control contexts; save exact development/confirmation partitions and reconstructed V14 predictions. Verify categorical unknowns and missing values, target/ID isolation, and train-only preprocessing. Test actual outer-fold size and all 3,493 fitting rows with 500 queries. Require finite predictions, repeated fresh-construction and serialized-state predictions below INR 0.01 maximum difference, and permutation/split/12 individual query differences no greater than INR 10 with labeled pilot RMSE drift no greater than INR 1. No rounding to satisfy tolerances. A deterministic 128-query inference cap and native autobatching control memory; they do not subsample fitting labels. Forecast remaining work from measurements with 30% contingency and protect ten minutes for delivery. Engineering has at most 30 minutes after the conservative task start; stop if infeasible. Repairs before development may correct implementation, without changing candidate or gates.

## Evaluation and decision

Use the existing five-fold assignments 42 and 71 with identical fitting/validation rows. Both assignments must improve ordinary INR RMSE and their mean relative gain must reach **0.20%**. Rank qualifying weights by mean gain, then smaller weight. Save the locked choice before opening confirmation. All original labels contribute to every primary score.

Confirm only that weight on 48211, 48212, 48213, stratified 27421 and canonical-city groups (25 folds). Every random assignment must improve and their mean relative gain must reach 0.20%. Require a positive lower bound of the same conditional paired-row 95% interval (5,000 draws, seed 8319), and no more than 1% degradation on either stress split. Report all results, tail errors and influential rows. Previously examined labels and partitions supply conditional evidence, not untouched evaluation or a hidden-test guarantee.

Only a fully qualified candidate proceeds to full-data inference and a fresh standalone notebook, with exact 500 IDs/order, headers, finite nonnegative values, formula verification, CSV round-trip, and INR 0.01 reproduction. Original V14 computational provenance and Mac fitted-state restrictions remain applicable. Otherwise preserve the byte-identical scored V14 CSV and notebook. A lower actual portal score is required to establish improvement; no portal upload is performed by this experiment. Release only its owned free T4 and verify release.

## Why this choice

See RESEARCH_SOURCES.md. The user has requested a new attempt after the previous campaign stop recommendation. That permits this bounded experiment; it does not create new labels or reset validation-selection bias. LimiX-2 is unmeasured locally and offers a distinct model prior. TabFM requires a separate context-sampling design; TabSTAR emphasizes text fields absent from this short-category schema; TabSwift emphasizes speed and compares against earlier models. None is proved inferior here. Retesting completed families or selecting an outlier correction from exhausted OOF labels has weaker justification.

## Local execution amendment before any predictions

Automatic approval review rejected transferring the dataset to Colab. The unused owned T4 was released and zero active assignments were verified. Keep all data local. Use the public MLX implementation in open PR 60 at immutable revision `cdf1fb7a54416c2886582c01715704d8560be4fe`; it is an unmerged contribution, not the official default runtime. Keep the same checkpoint, train-only adapter, two fixed members, single weight and decision gates. Verify complete checkpoint parameter coverage and record local numerical behavior; do not assert exact equivalence with PyTorch. Explicitly direct converted weights and caches into the V23 workspace. No local data or artifacts are uploaded.

Local feasibility context: Apple M1, 8 GiB total system memory. Limit MLX allocations to 3 GiB and its cache to 128 MiB. These are engineering limits, not altered labels or numerical tolerances. Stop on insufficient memory or an elapsed-time forecast that cannot fit.

## Explicit export permission and official CUDA retry

The user explicitly authorized the Colab dataset upload. The local full-size attempt passed support-preprocessing invariance but failed an 11.2 GB buffer allocation before a prediction completed. Preserve that evidence; it is not an accuracy failure. Return to the official Torch CUDA backend on the same selected free Colab account. Keep the two members, one blend, checkpoint, labels and all quality gates unchanged. Use a fixed 1024-query cap to avoid six repeated support passes per outer fold; native autobatching may tile computation without dropping fitting rows. There have been no V23 accuracy results. The one-hour hard stop remains 23:23:09 IST. Engineering may use up to 23:03:09, with training ending by 23:18:09 and five minutes reserved for delivery. These time allocations supersede the earlier internal allocations; the total ceiling is unchanged.

## Time-window clarification and singleton repair

Count the user's available hour from the timing message, using a conservative start of 22:38:30 IST (59 seconds before the first recorded post-message clock). Hard stop 23:38:30, engineering cutoff 23:08:30, training cutoff 23:33:30. The earlier research start and all previous authorization/source records remain in engineering_history; no previous experiment ledger changes. This corrects the initial overly conservative inclusion of research preceding the user's time constraint. The user-facing stop was stated as 23:38 IST.

The first CUDA pilot passed same-state reproduction and reached individual-query testing. The official predictor squeezes a singleton result to a scalar; normalize that output to a one-dimensional vector. No prediction values, quality gates, weights, model settings or labels change. No development accuracy result has been opened.
