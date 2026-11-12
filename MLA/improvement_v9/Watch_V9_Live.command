#!/bin/zsh
# Displays logs only. The training and delivery jobs run separately.
TASK_DIR="${0:A:h}"
print 'V9 recovery, confirmation and delivery logs'
print ''
print 'New lines appear when a fit, fold or delivery step reports progress.'
print 'Quiet periods during training are normal.'
print 'Press Control+C or close this window to stop viewing; training continues.'
print ''
exec /usr/bin/tail -n 12 -F "$TASK_DIR/reconfirm_portable_v7.log" "$TASK_DIR/colab_execute_portable_notebook.log" "$TASK_DIR/finalize_portable_delivery.log"
