# Accepted V15 implementation

User explicitly requested implementation on October 28, 2026. Eight hours of free T4 only, including setup and interruptions; two-hour delivery reserve. Preserve V14, portal RMSE 11,736,110.7877709.

Five fixed components: F40-T080, F40-T120, N32-T080, N32-T100, N32-T120. Exact V14 checkpoint/environment, eight estimators, seed 10242, float32, fixed 500-row inference adapter and original targets in INR millions. F40 is the unchanged V14 feature builder. N32 retains original predictors without target/ID, normalizes six categoricals exactly as V14, uses fold-local vocabularies and native parsed datetime; no derived features. Malformed dates become NaT.

V15 = .25*V9 + .75*((1-w)*P35_original + w*new_component), w in [.25,.5,.75,1]. Twenty comparisons, fifty development folds on assignments 42 and 71. Engineering pilots for all five precede development; no substitute models. Automatic temperature calibration disabled.

Require gains on both development assignments and mean relative gain >=.002 against V14. Rank by descending mean relative gain, lower w, then setting ID; freeze one candidate. Confirm on random 48211/48212/48213, stratified 27421 and canonical-city folds. Require all random gains positive, mean relative gain >=.002, paired-row 95% lower gain bound above zero (5000 draws, seed 8319), and city/stratified deterioration <=1%. No finalist means no confirmation/new CSV/new notebook: retain byte-identical V14.

Require fresh prediction/CSV reproduction <INR .01; permutation/split/individual behavior <=INR 10 and pilot RMSE change <=INR 1. Exact target/ID feature invariance. Complete hashed folds and independent score/decision reconstruction. Qualified export requires a fresh executed standalone notebook, 500 original IDs/order, exact headers, finite nonnegative predictions, CSV round-trip and malformed/cache rejection checks. No automatic portal upload. One manual trial only after qualification and reproduction.

Stage caps: audit/setup/pilot 1h, development 2.5h, confirmation 2.5h, delivery 2h. Forecast fits with 30% contingency before screening, retain reserve and stop if mandatory work cannot fit. Global eight-hour persisted wall clock; artifacts by October 29 21:00 IST, manual trial target 22:00 IST; live portal displayed October 29 23:59 with timezone unshown. No paid API, estimator/seed grids, target caps, label removal, additional families or fine-tuning. Historically examined labels are conditional robustness evidence.
