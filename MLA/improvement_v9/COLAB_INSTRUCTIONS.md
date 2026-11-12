# Run V9 in Google Colab

1. Download **V9_Research_Colab_Lite.ipynb** and **V9_Official_Inputs.zip** from this delivery. Extract the inputs ZIP: it contains the four official CSVs.
2. Open https://colab.research.google.com/ and choose **File → Upload notebook**. Select **V9_Research_Colab_Lite.ipynb**. The earlier `V9_TabPFN_Pilot.ipynb` is only an interrupted setup attempt.
3. Choose **Runtime → Change runtime type → T4 GPU → Save**. Use the free T4 option. No paid upgrade is needed.
4. Choose **Runtime → Run all**. Keep the tab open. The first cell installs pinned dependencies. When the file picker appears, upload all four extracted CSVs together: `Train_dataset.csv`, `Test_dataset.csv`, `Data_Dictionary.csv`, `Sample_Solution.csv`.
5. The notebook pilots both families, screens the available fixed settings, refines eligible candidates, confirms frozen finalists and reproduces V7. It may correctly finish with **retain V7**. A GPU pilot failure is reported explicitly and the notebook continues with EBM; it does not invent TabPFN results.
6. Allow the final **V9_research_results.zip** download. Send that ZIP back in this chat. If execution fails, send the full error text and the cell heading. Do not upload a CSV or submit the one-time coding assignment until the results have been reviewed.

Use this notebook directly; **do not upload V9_GPU_Input.zip**. That ZIP belonged to the interrupted automated transfer. No other project folders or earlier notebooks are required.

Current evidence: four CPU EBM settings completed locally, none qualified. TabPFN has not trained yet. The best confirmed portal score remains V7's **12,024,316.5355221**.

If Colab reports that a GPU is unavailable, stop and send that message. The screenshot estimate of runtime duration is not a guarantee. If the setup cell explicitly requests a restart after changing dependencies, restart the runtime and choose Run all again.

Qualified output folders contain a CSV and a matching coding notebook, freshly executed in a separate process. The final research ZIP includes input/source/output hashes, feasibility details, fold predictions and comparisons. The notebook does not log in to Olympus or upload the final coding assignment.

Browser freeze recovery: the original notebook had one 580,616-character embedded line. The Lite copy compresses the payload and keeps all code lines at most157 characters, with old outputs cleared. Model features, settings and folds are unchanged. If the page remains frozen after1–2 minutes, reopen Colab, interrupt any still-running old execution, and upload the Lite notebook. Restart the session before Run all; re-upload the four CSVs if they are absent.
