"""Small file-based RPC boundary: candidate libraries never enter V14's process."""
import contextlib
import json
import pickle
import re
import sys
import traceback
from pathlib import Path


def main():
    fitted = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with Path(request['input']).open('rb') as handle:
                payload = pickle.load(handle)
            with contextlib.redirect_stdout(sys.stderr):
                import candidate_models as models
                action = request['action']
                if action == 'inspect':
                    result = models.runtime()
                elif action == 'fit':
                    fitted = models.fit(**payload)
                    result = fitted['metadata']
                elif action == 'predict':
                    result = {'prediction': models.predict(fitted, payload), 'resources': models.resources()}
                elif action == 'save':
                    import joblib
                    joblib.dump(fitted, payload, compress=0)
                    result = {'saved': True}
                elif action == 'load':
                    import joblib
                    fitted = joblib.load(payload)
                    result = {'loaded': True}
                else:
                    raise ValueError('Unknown candidate action')
            with Path(request['output']).open('wb') as handle:
                pickle.dump(result, handle, protocol=5)
            status = {'ok': True}
        except BaseException as error:
            reason = re.sub(r'https?://\S+', '<url>', str(error))[:1800]
            print(re.sub(r'https?://\S+', '<url>', traceback.format_exc()), file=sys.stderr, flush=True)
            status = {'ok': False, 'error_type': type(error).__name__, 'reason': reason}
        print('MLA_RESPONSE ' + json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
