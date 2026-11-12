"""Independent artifact checks; no training, tuning, or root-file promotion."""
from pathlib import Path
import argparse, ast, builtins, csv, hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
STEM='submission'
EXPECTED_HEADERS=['Registration Number','Annual Turnover']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(condition,message):
    if not condition:raise AssertionError(message)

def independent_validate(frame,test):
    require(frame.shape==(500,2),'Expected 500 rows and two columns')
    require(frame.columns.tolist()==EXPECTED_HEADERS,'Headers or order are incorrect')
    ids=frame['Registration Number']
    require(ids.is_unique,'Duplicate restaurant IDs')
    require(ids.tolist()==test['Registration Number'].tolist(),'Restaurant IDs/order mismatch')
    require(pd.api.types.is_numeric_dtype(frame['Annual Turnover']),'Turnover must be numeric')
    values=frame['Annual Turnover'].to_numpy(float)
    require(np.isfinite(values).all(),'Predictions must be finite')
    require((values>=0).all(),'Predictions must be nonnegative')

def check_notebook(path):
    book=json.loads(path.read_text())
    require(book.get('nbformat')==4,'Expected notebook format 4')
    cells=book['cells'];code=[c for c in cells if c['cell_type']=='code']
    require(bool(code),'Notebook has no code cells')
    require(all(isinstance(c.get('execution_count'),int) and c['execution_count']>0 for c in code),'Some code cells are unexecuted')
    errors=[o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']
    require(not errors,'Notebook contains saved errors')
    require(any(c.get('outputs') for c in code),'Notebook contains no saved outputs')
    ids=[c['id'] for c in cells if 'id' in c]
    require(len(ids)==len(set(ids)),'Duplicate notebook cell IDs')
    cover='\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='markdown').casefold()
    for identity in ['submission','submission','submission']:
        require(identity.casefold() in cover,'Missing required cover identity: '+identity)
    # Static portability inspection. The runtime may need its normal pip libraries
    # and the source CSVs, but it must not import this workspace's helper modules.
    local_modules={p.stem for p in ROOT.parent.glob('improvement_v*/**/*.py')}
    imported=set();definitions=set();source='\n\n'.join(''.join(c['source']) for c in code)
    for i,c in enumerate(code):
        cell_source=''.join(c['source'])
        parsed=ast.parse(cell_source,filename=f'notebook_cell_{i}')
        compile(parsed,f'notebook_cell_{i}','exec')
        for n in ast.walk(parsed):
            if isinstance(n,ast.Import):
                for alias in n.names:
                    imported.add(alias.name.split('.')[0])
                    require(alias.name.split('.')[0] not in local_modules,'Notebook imports a workspace helper: '+alias.name)
            if isinstance(n,ast.ImportFrom):
                require(n.level==0,'Notebook has a relative module import')
                module=(n.module or '').split('.')[0];imported.add(module)
                require(module not in local_modules,'Notebook imports a workspace helper: '+module)
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):definitions.add(n.name)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['insert','append']:
                require(ast.unparse(n.func.value)!='sys.path','Notebook modifies sys.path for external code')
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['exec','eval']:
                require(not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr in ['read_text','read_bytes'] for v in ast.walk(n)),'Notebook executes external source files')
    require(bool(definitions),'Notebook has no embedded implementation definitions')
    return {'code_cells':len(code),'executed_code_cells':len(code),'saved_errors':len(errors),
            'output_items':sum(len(c.get('outputs',[])) for c in code),
            'static_self_containment':'No workspace helper imports, relative imports, sys.path modifications, or external-source execution; all Python cells compile.',
            'imported_modules':sorted(imported),'sha256':sha(path)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--csv-only',action='store_true');args=parser.parse_args()
    result={'passed':False,'checks':{},'failures':[],'candidate_portal_score':None}
    def check(name,fn):
        try:result['checks'][name]=fn()
        except Exception as exc:result['failures'].append({'check':name,'error':f'{type(exc).__name__}: {exc}'})
    test=pd.read_csv('~/Downloads/Test_dataset.csv')
    path=ROOT/(STEM+'.csv')
    submission=None
    def csv_check():
        nonlocal submission
        with path.open(newline='') as f:raw=list(csv.reader(f))
        require(len(raw)==501 and raw[0]==EXPECTED_HEADERS,'Raw CSV header/record count mismatch')
        require(all(len(r)==2 for r in raw),'Unexpected CSV field count')
        submission=pd.read_csv(path);independent_validate(submission,test)
        require((submission['Annual Turnover']>0).all(),'Candidate contains zero turnover')
        parts=np.load(ROOT/'results/full_prediction_parts.npz',allow_pickle=False)
        require(np.allclose(submission['Annual Turnover'],parts['candidate'],rtol=1e-12,atol=.001),'CSV differs from fitted prediction array')
        return {'rows':len(submission),'headers':list(submission),'unique_ids':int(submission['Registration Number'].nunique()),
                'matches_test_order':True,'finite_positive':True,'sha256':sha(path),
                'min':float(submission['Annual Turnover'].min()),'max':float(submission['Annual Turnover'].max())}
    check('csv',csv_check)
    def invalid_cases_check():
        require(submission is not None,'Candidate CSV unavailable')
        from pipeline import check_submission
        bad={
            'missing_row':submission.iloc[:-1],
            'extra_row':pd.concat([submission,submission.iloc[:1]],ignore_index=True),
            'reversed_rows':submission.iloc[::-1].reset_index(drop=True),
            'renamed_header':submission.rename(columns={'Annual Turnover':'Turnover'}),
            'reversed_headers':submission[EXPECTED_HEADERS[::-1]],
            'extra_index_column':submission.assign(index=0),
            'student_id_in_rows':submission.assign(**{'Registration Number':'submission'}),
        }
        b=submission.copy();b.loc[1,'Registration Number']=b.loc[0,'Registration Number'];bad['duplicate_id']=b
        for name,value in [('nan',np.nan),('positive_infinity',np.inf),('negative_infinity',-np.inf),('negative_turnover',-1)]:
            b=submission.copy();b.loc[0,'Annual Turnover']=value;bad[name]=b
        for name,b in bad.items():
            for validator in [independent_validate,check_submission]:
                rejected=False
                try:validator(b,test)
                except (AssertionError,ValueError,TypeError):rejected=True
                require(rejected,f'{validator.__name__} accepted invalid case {name}')
        return {'rejected_cases':list(bad),'validators':['independent_validate','pipeline.check_submission']}
    check('invalid_examples',invalid_cases_check)
    def backup_check():
        ledger=json.loads((ROOT/'confirmed_portal_result.json').read_text());records={}
        for kind in ['csv','ipynb']:
            item=ledger['files'][kind];backup=Path(item['file']);digest=sha(backup)
            require(digest==item['sha256'],'Confirmed v3 backup changed: '+kind)
            records[kind]={'sha256':digest,'matches_confirmed_ledger':True}
        old=pd.read_csv(ledger['files']['csv']['file']);independent_validate(old,test)
        parts=np.load(ROOT/'results/full_prediction_parts.npz',allow_pickle=False)
        require(np.allclose(old['Annual Turnover'],parts['previous'],rtol=1e-12,atol=.01),'Full refit fails to reproduce scored v3')
        records['portal_rmse']=ledger['latest_portal_rmse'];records['rank_at_reported_submission']=ledger['rank']
        return records
    check('confirmed_v3_backup',backup_check)
    if not args.csv_only:check('notebook',lambda:check_notebook(ROOT/(STEM+'.ipynb')))
    result['passed']=not result['failures'];result['scope']='CSV and backups only' if args.csv_only else 'CSV, backups and executed notebook'
    out=ROOT/'results/submission_checks.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1

if __name__=='__main__':sys.exit(main())
