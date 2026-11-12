# How to test the v7 candidates

1. Read REPORT.md for the measured evidence and first-place feasibility of each trial.
2. Upload only the CSV inside candidates/trial_1 to the restaurant-turnover Hackathon.
3. Record the returned RMSE with trial_1 and its exact CSV hash in portal_trials.csv.
4. Repeat with the separate CSV inside candidates/trial_2 if a second trial exists.
5. Keep the file with the lowest observed portal RMSE, including the preserved v5 file.
6. Submit only its matching executed notebook to LLJ_Oct_2026 after the final choice.

The trial CSVs have the same required filename in different folders. Keep the folders
separate so scores are associated with the right version. Upload individual CSVs to
the Hackathon, not the delivery ZIP or research notebook. The research notebook is
for rerunning experiments; the notebook beside each trial CSV reproduces that trial.

The original v5 remains backed up and was not replaced. No portal upload was made
by these scripts. An exploratory candidate has no known portal score until tested.
