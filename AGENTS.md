# Repository Working Agreements

- `main` is the normal integration branch. Commit compatible features and fixes directly to `main` after focused checks.
- Create a separate branch and worktree only when two implementations change the same behavior incompatibly or must run concurrently with different code.
- A Pull Request is optional for this repository; use one when review or conflict isolation is useful, not for every compatible change.
- Keep checkpoints, datasets, generated videos, logs, and large experiment artifacts outside Git.
- Preserve the pinned official Efficient-WAM source revision and record any deliberate revision change with the affected checkpoint.
