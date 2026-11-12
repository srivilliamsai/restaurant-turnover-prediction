# Using the verified V9 trial

**Confirmed current best: V9, RMSE 11,904,515.9864996; rank 9 in the October 27 screenshot.** Keep its matching CSV and executed notebook. The instructions below describe the original trial/reproduction procedure; the portal result has now been recorded.

The CSV and notebook are already executed and checked. You do not need to rerun Colab before testing this CSV.

1. Upload `submission/submission.csv` to the hackathon's mandatory Submission Data field.
2. Record the returned RMSE and the CSV SHA256 from `manifest.json`. Keep whichever observed file scores below the current V7 best of 12,024,316.5355221. A validation gain does not guarantee a portal gain.
3. Keep the matching notebook for the one-time coding assignment, and submit it only after the final model choice. The delivery ZIP is not a portal CSV.

## Optional Colab rerun

Open `submission/submission.ipynb`, choose a free T4 runtime and Run all. Upload the four official CSVs when prompted. Run every collapsed embedded-model cell. Package installation is followed by model execution in a fresh Python process, so it does not depend on variables from an earlier notebook run. The final cell downloads the CSV after all checks pass.

The notebook preserves your own fitted V7 checkpoint, which was freshly reproduced on the original Mac and verified in Colab. It trains the selected TabPFN component afresh on every run. It does not claim fresh V7 retraining on Linux. V7 training source is included. Exact dependency versions and checkpoint attribution are recorded in the notebook and environment manifest.

The original failed Colab reproduction is preserved separately. No acceptance tolerance or candidate setting was changed. No portal upload or paid compute was performed; the owned free T4 was released after its results were backed up.

Under the stated ordinary RMSE across the same 500 rows, this particular V9 candidate cannot reach the supplied 4.7967M leader score. It is an incremental trial whose actual score must be measured on the portal.
