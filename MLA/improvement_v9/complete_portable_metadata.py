"""Attach recorded execution metadata and concise usage to the verified delivery."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
run=root/'colab_cli_runs/20261026T160148Z'
out=run/'portable_delivery'
status=json.loads((out/'delivery_status.json').read_text())
assert status['status']=='completed'
assert json.loads((out/'independent_review.json').read_text())['status']=='PASS'
cleanup=json.loads((run/'runtime_cleanup.json').read_text())
assert cleanup['owned_runtime_released']
for name in ['colab_environment_manifest.json','runtime_cleanup.json']:
    shutil.copyfile(run/name,out/name)
stem='submission'
(out/'READ_ME_FIRST.md').write_text(f'''# Using the verified V9 trial

The CSV and notebook are already executed and checked. You do not need to rerun Colab before testing this CSV.

1. Upload `submission/{stem}.csv` to the hackathon's mandatory Submission Data field.
2. Record the returned RMSE and the CSV SHA256 from `manifest.json`. Keep whichever observed file scores below the current V7 best of 12,024,316.5355221. A validation gain does not guarantee a portal gain.
3. Keep the matching notebook for the one-time coding assignment, and submit it only after the final model choice. The delivery ZIP is not a portal CSV.

## Optional Colab rerun

Open `submission/{stem}.ipynb`, choose a free T4 runtime and Run all. Upload the four official CSVs when prompted. Run every collapsed embedded-model cell. Package installation is followed by model execution in a fresh Python process, so it does not depend on variables from an earlier notebook run. The final cell downloads the CSV after all checks pass.

The notebook preserves your own fitted V7 checkpoint, which was freshly reproduced on the original Mac and verified in Colab. It trains the selected TabPFN component afresh on every run. It does not claim fresh V7 retraining on Linux. V7 training source is included. Exact dependency versions and checkpoint attribution are recorded in the notebook and environment manifest.

The original failed Colab reproduction is preserved separately. No acceptance tolerance or candidate setting was changed. No portal upload or paid compute was performed; the owned free T4 was released after its results were backed up.

Under the stated ordinary RMSE across the same 500 rows, this particular V9 candidate cannot reach the supplied 4.7967M leader score. It is an incremental trial whose actual score must be measured on the portal.
''')
manifest_path=out/'manifest.json'
manifest=json.loads(manifest_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for relative,expected in manifest['files'].items():assert sha(out/relative)==expected,relative
for name in ['colab_environment_manifest.json','runtime_cleanup.json','READ_ME_FIRST.md']:
    manifest['files'][name]=sha(out/name)
manifest['owned_free_T4_released']=True
manifest['reproduction_protocol']={'V7':'separate fresh Mac fit, verified frozen-state inference in Colab','TabPFN':'fresh full-data fit in standalone notebook process','maximum_accepted_prediction_difference_INR':.01}
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
archive=root/'V9_Portable_Delivery.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for relative in sorted(manifest['files']):z.write(out/relative,relative)
    z.write(manifest_path,'manifest.json')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
status['archive_sha256']=sha(archive)
status['owned_runtime_released']=True
(out/'delivery_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'status':'completed','csv_sha256':manifest['files'][f'submission/{stem}.csv'],'package_sha256':sha(archive),'package_bytes':archive.stat().st_size},indent=2))
