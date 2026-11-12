# V10 implementation progress

Updated October 27, 2026, after the user switched the Colab CLI login.

The best observed portal result remains **V9: 11,904,515.9864996**. The user's latest screenshot shows current rank **11**. V10 completed all local qualification and reproduction checks, but its sole portal trial scored **11,925,736.5011738**, worse by **21,220.5146742** (0.1783%). The portal retained V9's best score. Recommend V9 and its matching notebook; no additional V10 upload is recommended.

## Current execution

- The user approved a **24-hour total cap**, using free compute only. The six settings, 24 development blends, and qualification requirements remain unchanged.
- The prior CLI authorization was revoked and its saved local login removed at the user's request. The user completed Google's login flow in their own Terminal. No credentials are included in artifacts.
- The resumed CLI connection succeeded, reported zero paid compute units and no active assignments, and then obtained a free T4 GPU. Pinned dependency installation and a clean-kernel restart are complete. Exact input, source, dependency, checkpoint and execution-contract checks passed.
- All **70 development folds** are complete and their saved prediction-file hashes have been checked. All six representations and 24 declared blends were screened. The backup watcher finished after delivery.
- The frozen finalist is C at replacement weight 100%, with 42 new-component features. All 50 confirmation folds are complete, all five gates passed, and the original-Mac city comparison passed. The mean random RMSE improvement is INR 4,083.91; the conditional bootstrap interval includes zero.
- Independent metric reconstruction and real P-component invariance checks passed. The 29-code-cell standalone notebook executed successfully and reproduced the final CSV within INR 0.000000007451. V9 reproduction passed separately.
- Final CSV checks passed for 500 rows, original unique IDs/order, required headers and finite nonnegative predictions. Its SHA256 is f87948ad9025872e4701fa970a1d84eefd86aa8c3cacec7705b813ccfa11ec1b.
- The complete delivery archive is V10_Delivery.zip. Final accounting preserves earlier charges and totals 16.9253 hours under the human-authorized 24-hour cap. The owned T4 was released; the CLI confirmed zero active assignments afterward.
- The finite coordinator completed and released its owned runtime. The original pre-portal archive is preserved separately; the updated delivery contains the screenshot, exact trial record and recommended matching V9 files.

## Preserved evidence

- **70 development fold pairs** are backed up locally: 10 matching V9 controls and 10 folds for each of the six new representations.
- C's selected declared blend, replacement weight 100%, improves development RMSE by INR 4,925.06 on assignment 42 and INR 3,600.48 on assignment 71. These are small development improvements, not confirmation or portal scores. The complete selection was frozen before confirmation.
- Both development control assignments were checked against source, feature-map, vocabulary and prediction hashes. Stable-batch versus original V9 RMSE differences are +INR 0.069916 (42) and +INR 0.063143 (71). Original OOF witnesses remain preserved.
- The passing foreground pilot took 1,040.03 seconds. Repeat-fit, split-batch and individual-row differences were INR 0.00; peak allocated GPU memory was 611,609,088 bytes. Target/ID isolation, permutation, missing and unseen input checks passed.
- All local feature, pruning, vocabulary/map and cache-corruption checks pass. Notebook structure checks and fresh GPU notebook reproduction passed, including all 29 executed code cells and verification of the embedded V7 state.

## Interruptions and numerical handling

- The first native T4 pilot failed batch invariance at INR 1.2951331064 and stopped before screening. A fixed 500-row internal-batch adapter now pads with a frozen training-feature row, discards padding predictions, and applies no rounding. The 500-row competition test input retains its original prediction shape.
- Two earlier detached Colab runtimes expired. Later, the foreground screen's local interactive transport stopped, leaving the 26 saved folds above. Unknown intervals were conservatively charged; no missing results were invented.
- Three bounded free-T4 allocation attempts on the earlier login returned Service Unavailable. That coordinator stopped before the account switch. Retry time remains charged.
- Budget history and the explicit amendment preserve earlier charges. The global total is the prior charged seconds plus the active epoch seconds; restarting a runtime or switching accounts never resets it.

## Remaining work

The single declared V10 portal trial is complete. The user-provided screenshot and score-to-file provenance are recorded in portal_trials/v10_trial_1.json and in the updated delivery. The portal's lower observed V9 score remains recommended. No V10 tuning or additional portal trial follows this result.

The one-time coding upload stays pending until the final model choice. The deadline timezone was not independently verified; no timezone confirmation is claimed.

Live local files:

- `transport_run/coordinator.log`
- `transport_run/status.json`
- `colab_results/run.log` after checkpoint restoration and backup start
- `colab_results/results/screening_progress.json` after screening resumes

Transport credentials remain in the CLI's normal credential store and are never printed or packaged. No paid resources are requested.
