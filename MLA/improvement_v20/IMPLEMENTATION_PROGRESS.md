# V20 implementation progress

- Selected account verified with zero pre-existing active assignments; one owned free T4 allocated.
- V14 control dependency installation completed.
- Original-Mac V7 pilot reproduced its frozen witness within INR 1.49e-8.
- Nine local contract tests passed.
- First isolated environment creation failed because Colab lacks the ensurepip bootstrap. Retrying with a venv without pip and the supported external pip interpreter option; no system packages are inherited.
- No V20 accuracy results inspected. Setup, failed attempts and recovery all count toward the original six-hour ledger.

- Both isolated candidate environments now installed successfully on T4, with Torch 2.11.0+cu130 and no discoverable cuDF or Numba.
- Official TabICL regression checkpoint immutable revision/hash and installed source snapshots locked before pilots. Kumo retains the previously inspected immutable source/checkpoint.

- Before any accuracy testing, audit rejected a budget-guard bug: the 1-hour engineering ceiling was applied before subtracting the 1.5-hour delivery reserve, incorrectly leaving zero time. Corrected by applying engineering and global limits independently, and added a regression check. The failed coordinator is collecting available logs and releasing its owned runtime. Original execution clock is unchanged.

- Before any accuracy test, source inspection confirmed TabICL mean output and its all-NaN query-column masking. Preserved the original V11 fixed-500-row adapter with training-feature-only padding chosen by maximum observed fields. This is the original recipe, not an accuracy-selected amendment.

- Replacement runtime restored the exact locked candidate environments successfully in 285.16 seconds. Final source freeze updated before remote audit/pilot. All earlier time remains charged.

- All three full-sized engineering pilots passed. I8 fold15.42s; Kumo Small fold19.92s; I16 fold22.22s. Both control folds reproduced exactly.
- All settings admitted before development scores; forecast12.47min development and99.95min maximum confirmation including original-platform controls, with 30% contingency. Development started at original elapsed45.78min. Computational sources are frozen.

- Development complete: all30 candidate folds and20 fixed blends. Best I16@10% mean gain0.002639%, with a loss on assignment71; no finalist qualified.
- Confirmation and a new submission notebook/CSV were skipped according to the frozen decision. Exact scored V14 files were preserved.
- Independent reconstruction PASS for every fold/context/target/source/hash and all20 comparisons; maximum metric difference INR3.73e-9.
- Owned T4 released; fresh selected-account check found zero active assignments. No paid compute or portal upload.
