# V14 engineering evidence

The two V11 exclusions were resolved sufficiently to complete genuine accuracy experiments in V14. All numerical sources remained frozen during development and confirmation.

## TabPFN-3.5

The downloaded artifact was copied to a regular local file retaining the `.safetensors` suffix. Its parsed header contained 595 tensor entries and the architecture/configuration metadata. The installed official Checkpoint loader recognized this format. The fitted model reported `tabpfn.architectures.tabpfn_v3_5.TabPFNV3p5` and `TabPFNV3p5Config`. No 2.5 fallback was used.

- Official repository: Prior-Labs/tabpfn_3_5
- Revision: `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`
- File: `tabpfn-v3.5-20260909.safetensors`
- SHA256: `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`
- Installed checkpoint-loader SHA256: `f7c86363ccc96af4ffe10ec23221ed90dfea304f04b6c4e48330ba25bbf807de`
- Full-size pilot fold: 22.19 seconds; every recorded permutation/batch/individual difference and fresh-repeat difference was INR 0.

The exact V11 failure path cannot be established from its truncated saved exception. The successful V14 named-file path, loader source identity, format inspection and actual architecture establish what worked in this run. A unique historical root cause is not claimed.

## TabICL precision

The full 699-row pilot passed the predeclared INR 10 behavior limit. Maximum permutation difference was INR 7.62939453125; the corresponding ordinary RMSE change was INR 0.02742534. Fresh same-layout fitting matched exactly. All twelve individual probes and missing/unseen-feature probes passed.

I8 completed all ten development folds and its eight declared route/weight comparisons. Its best mean gain, 0.07620%, did not meet the fixed 0.20% development requirement. It was excluded on measured accuracy after successful engineering, rather than being counted as an accuracy failure from the old precision check.

## Download-cache handling

A bounded operational helper waits for successful standalone inference before moving that worker's downloaded pretrained-weight cache outside the delivery directory. This is a reversible packaging operation; model source, parameters and predictions are unchanged. Official weight revisions and hashes remain recorded and the standalone notebook retrieves them when needed. The completion record is packaging_cache_relocation.json.
