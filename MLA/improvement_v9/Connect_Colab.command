#!/bin/zsh
# One-time Google sign-in. Only lists sessions; never creates or stops a runtime.
set -u
umask 077

TASK_DIR="${0:A:h}"
COLAB_BIN="$TASK_DIR/colab_cli_env/bin/colab"

print 'Connect Google Colab to the V9 workflow'
print ''
print '1. Open the Google sign-in URL printed below (Command-click it).'
print '2. Use the same Google account as your Colab notebook.'
print '3. Review Google’s access request and approve it if you agree.'
print '4. Paste Google’s authorization code into THIS Terminal window.'
print '   Keep the code and your password out of chat.'
print ''
print 'This command only lists sessions. It does not buy compute or launch training.'
print ''

if "$COLAB_BIN" --auth oauth2 sessions; then
    print ''
    print 'Session check completed. Return to Codex and say: Colab connected.'
else
    print ''
    print 'Connection did not complete. Tell Codex the error message, without any codes or tokens.'
fi
print ''
read -r '?Press Return to close this window. '
