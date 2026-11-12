"""CPU-only smoke checks. Synthetic branch tests are never training results."""
import ast,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from core import ROOT,Engine,save,sha
import features,modeling,workflow,verify,build_delivery

def main():
    for p in list(ROOT.glob('*.py'))+list((ROOT/'transport').glob('*.py')):compile(p.read_text(),str(p),'exec')
    # Compile the actual rendered remote templates, including inserted supervisor.
    tree=ast.parse((ROOT/'colab_control.py').read_text());templates=0
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='execute' or not node.args:continue
        value=node.args[0]
        if isinstance(value,ast.Constant) and isinstance(value.value,str):compile(value.value,'<remote template>','exec');templates+=1
    source=(ROOT/'colab_control.py').read_text()
    for stage in ['audit','pilot','screen','confirm','export']:
        marker='''            execute("""import os,sys,json,subprocess''';a=source.index(marker)+len('            execute("""');b=source.index('""".replace(\'STAGE\'',a)
        rendered=source[a:b].replace('STAGE',repr(stage)).replace('SUPERVISOR',repr((ROOT/'transport/stage_supervisor.py').read_text()))
        compile(rendered,'<rendered foreground '+stage+'>','exec');templates+=1
    config=ROOT/'resolved_models.json';assert not config.exists(),'Do not replace a real frozen execution profile'
    save(config,{'preflight_only':True})
    try:
        with tempfile.TemporaryDirectory(prefix='preflight_',dir=ROOT) as tmp:
            e=Engine(ROOT.parent,Path(tmp));verify.feature_checks(e)
            pilot={'status':'PASS','settings':{n:{'status':'SYNTHETIC_BRANCH_ONLY'} for n in features.SPECS},'screen_settings':list(features.SPECS),'sources':e.sources,'inputs':e.inputs,'model_contract_sha256':sha(config),'model_contract':{'preflight_only':True}}
            save(e.out/'pilot.json',pilot);save(e.out/'experiment_plan.json',{'preflight_only':True})
            # Equal-to-control predictions must produce no finalists, no Mac fits.
            def equal(name,scheme):
                v7,t9=e.reference_parts(scheme);return t9 if name.startswith('T') else .5*v7+.5*t9
            e.component=equal
            workflow.screen(e);workflow.confirm(e)
            d=json.loads((e.out/'decision.json').read_text());assert d['complete'] and not d['qualified'] and d['choices']==[]
            # Explicit zero-floor and T/R formula checks, independent arithmetic.
            a=np.array([10.,20.]);b=np.array([30.,40.]);c=np.array([50.,60.])
            np.testing.assert_allclose(modeling.combine(a,b,c,'T32',.25),.5*a+.5*(.75*b+.25*c))
            np.testing.assert_allclose(modeling.combine(a,b,c,'R1',.25),.75*(.5*a+.5*b)+.25*c)
            assert (modeling.combine(a,b,np.full(2,-1e3),'R1',1)==0).all()
            # The actual coordinator's no-finalist branch must never create or poll
            # a Mac/confirmation process. Execute its real branch with inert hooks.
            coordinator=(ROOT/'transport/continue_finite.py').read_text();a=coordinator.index(" if lock['choices']:");b=coordinator.index(" control('collect',timeout=600)\n if not (out/'decision.json')",a)
            fragment='if True:\n'+coordinator[a:b]
            class NoProcesses:
                def Popen(self,*args,**kwargs):raise AssertionError('No-finalist branch launched a process')
            exec(compile(fragment,'<real no-finalist coordinator branch>','exec'),{'lock':{'choices':[]},'subprocess':NoProcesses()})
            # Build and compile the real standalone notebook without executing/fitting.
            inventory=ROOT/'source_inventory.json'
            if not inventory.exists():save(inventory,{'files':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('*.py')}})
            candidate=e.out/'notebook';candidate.mkdir();(candidate/'submission.csv').write_bytes((ROOT/'reference/scored_v9.csv').read_bytes())
            choice={'name':'T32','family':'T','weight':.25,'inference_cost':32}
            book,path,manifest=build_delivery.build_notebook(e,{'qualified':True,'choice':choice},candidate)
            for cell in book.cells:
                if cell.cell_type=='code':compile(cell.source,'<standalone cell>','exec')
            count=sum(c.cell_type=='code' for c in book.cells)
            record={'status':'PASS','gpu_model_fits':0,'actual_cli_audit_previously_passed':True,'feature_and_cache_checks':True,'no_finalist_actual_branch':True,'both_blend_formulas':True,'remote_templates_compiled':templates,'standalone_code_cells_compiled':count,'standalone_numerical_execution':False,'sources':e.sources,'inputs':e.inputs}
            save(ROOT/'local_preflight.json',record);print(json.dumps(record,indent=2))
    finally:config.unlink(missing_ok=True)

if __name__=='__main__':main()
