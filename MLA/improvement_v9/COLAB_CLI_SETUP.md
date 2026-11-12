# Colab CLI connection

Status on October 26: Google sign-in succeeded. A standard free Tesla T4 was
allocated through the CLI, and the full-fold TabPFN feasibility checks passed.
The active V9 run is recorded in `CLI_PROGRESS.md`; another sign-in or a
duplicate browser run is unnecessary while that run is active.

Google's official `google-colab-cli==0.7.5` is installed in the isolated
`colab_cli_env` folder. `colab --help` and `colab version` were verified locally.
The existing model environments and scored V7 files are unchanged.

## One-time sign-in

Double-click `Connect_Colab.command` in Finder. In its Terminal window, open the
Google authorization link, sign in to the account used for Colab, review the
requested access, and paste the resulting authorization code into that Terminal
window. Do not paste passwords, authorization codes or tokens into chat.

Google's CLI requests account identity, Colab, Google Cloud and app-selected
Drive-file access. This sign-in is Google's requirement for accessing account
sessions, not a subscription or a compute purchase. Credentials are stored by
the official CLI under `~/.config/colab-cli/`, outside the research delivery.

The launcher only lists sessions. After sign-in, return to Codex and say
“Colab connected.” We must inspect existing sessions and running work before
starting another job.

## Cost and limits

The official CLI and Colab MCP server are open-source. They do not supply a
separate free GPU allocation or extend Colab's runtime limits. Actual access for
this account must be checked after sign-in. Use available free resources only;
do not upgrade, buy units or request premium hardware. No SSH or remote desktop
workflow is needed.

## Sources

- [Google's CLI announcement](https://developers.googleblog.com/introducing-the-google-colab-cli/)
- [Official CLI source and instructions](https://github.com/googlecolab/google-colab-cli)
- [Official Colab MCP source](https://github.com/googlecolab/colab-mcp)
- [Colab resource limits](https://research.google.com/colaboratory/faq.html)
