# START_BOT Checkout Model

`START_BOT.bat` runs the checkout that contains the batch file. The desktop
shortcut currently targets the Windows checkout:

```text
C:\Users\joshd\canompx3\START_BOT.bat
```

Codex often works in the WSL checkout:

```text
/home/joshd/canompx3
```

Those are separate Git working trees. A commit pushed from the WSL checkout does
not change the Windows checkout until the change is merged, pulled, or ported
there. If the desktop shortcut still looks old after Codex pushed work, first
check the startup banner printed by `START_BOT.bat`:

```text
[Repo] C:\Users\joshd\canompx3
[Repo] This shortcut runs the Windows checkout above.
[Repo] WSL/Codex branch pushes do not change this app until merged or pulled here.
```

Interpretation:

- A Codex branch such as `codex/opportunity-awareness-overlay` will not affect
  the shortcut while the shortcut checkout remains on `main`.
- To inspect the exact branch and commit, run these from the Windows checkout:

```bat
cd /d C:\Users\joshd\canompx3
git branch --show-current
git rev-parse --short HEAD
git status --short --branch
```

The batch launcher intentionally does not run these Git commands itself. It
must keep double-click startup reliable even if Git is slow, missing from PATH,
or the checkout has unusual upstream state.

Operator rule: when a dashboard/UI change is meant to affect the desktop
shortcut immediately, either merge/pull it into `C:\Users\joshd\canompx3` or
port the specific dashboard patch onto that checkout. Do not assume that a
pushed Codex branch has changed the shortcut target.
