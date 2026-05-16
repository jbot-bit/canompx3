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
[Repo] branch=<branch> commit=<commit> upstream=<upstream> ahead=<n> behind=<n>
```

Interpretation:

- `branch` and `commit` are the exact code the shortcut is serving.
- `behind > 0` means the shortcut checkout does not include newer upstream
  commits.
- A Codex branch such as `codex/opportunity-awareness-overlay` will not affect
  the shortcut while the shortcut checkout remains on `main`.

Operator rule: when a dashboard/UI change is meant to affect the desktop
shortcut immediately, either merge/pull it into `C:\Users\joshd\canompx3` or
port the specific dashboard patch onto that checkout. Do not assume that a
pushed Codex branch has changed the shortcut target.
