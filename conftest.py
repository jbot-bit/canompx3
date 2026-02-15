"""Root conftest: prevent pytest from traversing symlinks/mount points."""

# Block pytest from stat()-ing the local_db symlink (WinError 448)
collect_ignore = ["local_db"]
collect_ignore_glob = ["local_db/*"]
