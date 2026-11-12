"""Submission contract tests, including intentionally invalid examples."""
import csv, io, json, math, unittest, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
STEM='submission'
HEADERS=['Registration Number','Annual Turnover']

def validate_csv(text, expected_ids):
    reader=csv.DictReader(io.StringIO(text))
    assert reader.fieldnames==HEADERS, 'Incorrect headers or an unwanted index column'
    rows=list(reader)
    assert len(rows)==500, 'Exactly 500 prediction rows are required'
    assert all(set(row)==set(HEADERS) for row in rows), 'Malformed CSV row'
    ids=[row[HEADERS[0]] for row in rows]
    assert len(set(ids))==500, 'Duplicate restaurant ID'
    assert ids==expected_ids, 'Restaurant IDs must match Test_dataset.csv in original order'
    values=[float(row[HEADERS[1]]) for row in rows]
    assert all(math.isfinite(v) and v>=0 for v in values), 'Invalid predicted turnover'
    return values

class SubmissionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('~/Downloads/Test_dataset.csv',newline='') as f:
            cls.ids=[r['Registration Number'] for r in csv.DictReader(f)]
        cls.text=(ROOT/(STEM+'.csv')).read_text()
        cls.rows=list(csv.reader(io.StringIO(cls.text)))

    def changed(self, mutate):
        rows=[r.copy() for r in self.rows]; mutate(rows)
        out=io.StringIO();csv.writer(out).writerows(rows);return out.getvalue()

    def test_delivered_csv(self): validate_csv(self.text,self.ids)
    def test_wrong_row_count_rejected(self):
        with self.assertRaises(AssertionError): validate_csv(self.changed(lambda r:r.pop()),self.ids)
    def test_wrong_header_rejected(self):
        with self.assertRaises(AssertionError): validate_csv(self.text.replace('Annual Turnover','Turnover',1),self.ids)
    def test_duplicate_id_rejected(self):
        with self.assertRaises(AssertionError): validate_csv(self.changed(lambda r:r[2].__setitem__(0,r[1][0])),self.ids)
    def test_permuted_rows_rejected(self):
        def swap(r):r[1],r[2]=r[2],r[1]
        with self.assertRaises(AssertionError): validate_csv(self.changed(swap),self.ids)
    def test_nonfinite_negative_and_text_rejected(self):
        for bad in ['nan','inf','-1','abc','']:
            with self.subTest(value=bad),self.assertRaises((AssertionError,ValueError)):
                validate_csv(self.changed(lambda r:r[1].__setitem__(1,bad)),self.ids)
    def test_extra_column_rejected(self):
        with self.assertRaises(AssertionError): validate_csv(self.changed(lambda r:[row.insert(0,'index') for row in r]),self.ids)
    def test_student_id_rejected(self):
        with self.assertRaises(AssertionError): validate_csv(self.changed(lambda r:r[1].__setitem__(0,'submission')),self.ids)
    def test_notebook_sources_and_outputs(self):
        import nbformat
        nb=nbformat.read(ROOT/(STEM+'.ipynb'),as_version=4);nbformat.validate(nb)
        for i,cell in enumerate(nb.cells):
            if cell.cell_type=='code':
                compile(cell.source,f'cell{i}','exec')
                self.assertIsNotNone(cell.execution_count)
                self.assertFalse(any(o.output_type=='error' for o in cell.outputs))
        cover=nb.cells[0].source
        self.assertIn('submission',cover)
        self.assertIn('submission',cover)
        self.assertIn('submission',cover)

if __name__=='__main__':
    results=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SubmissionContract))
    info=dict(tests_run=results.testsRun,failures=len(results.failures),errors=len(results.errors),
              csv_sha256=hashlib.sha256((ROOT/(STEM+'.csv')).read_bytes()).hexdigest(),
              notebook_sha256=hashlib.sha256((ROOT/(STEM+'.ipynb')).read_bytes()).hexdigest())
    (ROOT/'improvement_v3/results/submission_checks.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))
    if not results.wasSuccessful():raise SystemExit(1)
