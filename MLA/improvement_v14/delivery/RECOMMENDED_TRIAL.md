# Verified V14 portal result

The user supplied a screenshot of the V14 manual portal trial: RMSE **11,736,110.7877709**, previous best **11,904,515.9864996**, and displayed leaderboard rank **9**. V14 is now the best observed submission, improving V9 by **168,405.1987287 (1.4146328916%)**. Evidence: `portal_result.png`.

Retain `submission.csv` and its matching executed coding notebook, `submission.ipynb`. The one-time coding upload remains unconfirmed; no further V14 CSV trial is recommended.

Model: 25% scored V9 model + 75% actual TabPFN-3.5 Base, eight estimators, unchanged V9 features, all original targets.

Mean random confirmation RMSE reduction: 0.5364% (INR 106,031.34). City and stratified checks improved. Conditional paired-row 95% gain interval: INR 35,922.47–181,795.22. These local validation results are separate from the actual portal result above.

Fresh standalone reproduction maximum difference: INR 7.450580596923828e-09. All 29 code cells executed without errors. Full-blend permutation, target/ID, split-batch and individual-row differences were zero.

CSV SHA256: `39d49f79beb8549d04ffc78d134aad2407dc5e73914dc3c30e8bb162a3c0ba05`.

The returned score is recorded in `portal_trial.json`. The user identifies this trial as V14; the screenshot confirms the score and rank but does not display a CSV hash. The hash above identifies the verified local V14 candidate.

The free T4 was released; account inspection reported zero active assignments and zero compute-unit balance. The portal upload was performed by the user; the assistant performed no portal upload.
