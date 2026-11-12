#!/bin/zsh
set -u
TASK_DIR="${0:A:h}"
print 'V12 live model log — free Colab T4 through CLI'
print 'This window only reads backed-up logs; it does not launch another run.'
print 'Press Control-C to stop watching. Training continues independently.'
print ''
exec /usr/bin/tail -n 35 -F "$TASK_DIR/colab_results/run.log"
