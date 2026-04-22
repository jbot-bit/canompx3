#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
PRIMARY_ROOT="$(git -C "$PROJECT_ROOT" worktree list --porcelain | awk '/^worktree /{print $2; exit}')"

CODEX_BIN="$(command -v codex 2>/dev/null || true)"
if [[ -z "$CODEX_BIN" ]]; then
  echo "ERROR: codex CLI not found" >&2
  exit 2
fi

ensure_shared_venv_link() {
  if [[ -e "$PROJECT_ROOT/.venv-wsl" ]]; then
    return
  fi
  if [[ -n "${PRIMARY_ROOT:-}" && -d "$PRIMARY_ROOT/.venv-wsl" ]]; then
    ln -s "$PRIMARY_ROOT/.venv-wsl" "$PROJECT_ROOT/.venv-wsl"
  fi
}

ensure_private_context_links() {
  local source_root="${PRIMARY_ROOT:-}"
  if [[ -z "$source_root" || "$source_root" == "$PROJECT_ROOT" ]]; then
    return
  fi

  local name
  for name in SOUL.md USER.md MEMORY.md HEARTBEAT.md; do
    if [[ ! -e "$PROJECT_ROOT/$name" && -e "$source_root/$name" ]]; then
      ln -s "$source_root/$name" "$PROJECT_ROOT/$name"
    fi
  done

  if [[ ! -e "$PROJECT_ROOT/memory" && -e "$source_root/memory" ]]; then
    ln -s "$source_root/memory" "$PROJECT_ROOT/memory"
  fi
}

resolve_python() {
  ensure_shared_venv_link
  if [[ -x "$PROJECT_ROOT/.venv-wsl/bin/python" ]]; then
    printf '%s\n' "$PROJECT_ROOT/.venv-wsl/bin/python"
    return
  fi

  if [[ -n "${PRIMARY_ROOT:-}" && -x "$PRIMARY_ROOT/.venv-wsl/bin/python" ]]; then
    printf '%s\n' "$PRIMARY_ROOT/.venv-wsl/bin/python"
    return
  fi

  command -v python3 2>/dev/null || command -v python 2>/dev/null || true
}

PY="$(resolve_python)"
if [[ -z "${PY:-}" ]]; then
  echo "ERROR: python interpreter not found" >&2
  exit 2
fi

WORKSTREAM="${CANOMPX3_CODEX_DISCOVERY_WORKSTREAM:-mnq-hiroi-scan}"
BRANCH="${CANOMPX3_CODEX_DISCOVERY_BRANCH:-wt-codex-${WORKSTREAM}}"
WORKTREE="${CANOMPX3_CODEX_DISCOVERY_ROOT:-/tmp/canompx3-${WORKSTREAM}}"
PROFILE="${CANOMPX3_CODEX_DISCOVERY_PROFILE:-canompx3_search}"
PURPOSE="${CANOMPX3_CODEX_DISCOVERY_PURPOSE:-MNQ autonomous bounded discovery hub}"
TASK_TEXT="${CANOMPX3_STARTUP_TASK:-Autonomous bounded MNQ discovery hub: refresh the board stack, rebuild the ranked hiROI frontier, choose one honest queued candidate, use the cheap gate before discovery writes, verify durable changes, and continue only when the frontier still has an honest next step.}"
RESUME_MODE="${RESUME_MODE:-never}"

MAX_ITERS="${1:-0}"
AUTO_COMMIT="${AUTO_COMMIT:-1}"
AUTO_PUSH="${AUTO_PUSH:-0}"
AUTO_PR="${AUTO_PR:-0}"
AUTOSAVE_PATCH="${AUTOSAVE_PATCH:-1}"
STOP_FILE="${STOP_FILE:-$PROJECT_ROOT/mnq_autonomous_discovery.stop}"
LOG_ROOT="${LOG_ROOT:-/tmp/canompx3-mnq-autodiscovery}"
PROMPT_TEMPLATE="$PROJECT_ROOT/scripts/infra/codex-mnq-autonomous-discovery-prompt.md"
SCHEMA_PATH="$PROJECT_ROOT/scripts/infra/codex-mnq-autonomous-discovery-result.schema.json"
BOARD_STACK="$PROJECT_ROOT/scripts/tools/run_mnq_discovery_board_stack.sh"
FRONTIER_TOOL="$PROJECT_ROOT/scripts/tools/build_mnq_discovery_frontier.py"
CAPSULE_TOOL="$PROJECT_ROOT/scripts/tools/render_mnq_discovery_capsule.py"
TASK_ROUTE="$PROJECT_ROOT/scripts/tools/task_route_packet.py"
MOUNT_GUARD="$PROJECT_ROOT/scripts/tools/wsl_mount_guard.py"
SESSION_PREFLIGHT="$PROJECT_ROOT/scripts/tools/session_preflight.py"
STATE_FILE="$WORKTREE/.session/mnq_autonomous_discovery_state.json"
FRONTIER_FILE="$WORKTREE/.session/mnq_discovery_frontier.json"
FRONTIER_LEDGER="$WORKTREE/.session/mnq_discovery_frontier_ledger.json"
CAPSULE_FILE="$WORKTREE/.session/mnq_discovery_capsule.md"
CODEX_RUNTIME_HOME="${CANOMPX3_CODEX_RUNTIME_HOME:-$WORKTREE/.session/codex_home}"
LOCK_DIR="$WORKTREE/.session/mnq_autonomous_discovery.lock"
LOCK_PID_FILE="$LOCK_DIR/pid"

mkdir -p "$LOG_ROOT"

usage() {
  cat <<EOF
Usage:
  bash scripts/tools/mnq_autonomous_discovery_loop.sh [max_iters]

Environment:
  CANOMPX3_CODEX_DISCOVERY_WORKSTREAM   Managed workstream name (default: mnq-hiroi-scan)
  CANOMPX3_CODEX_DISCOVERY_BRANCH       Branch to reuse/create
  CANOMPX3_CODEX_DISCOVERY_ROOT         Worktree path
  CANOMPX3_CODEX_DISCOVERY_PROFILE      Codex profile (default: canompx3_search)
  AUTO_COMMIT                           1/0 auto-commit verified durable changes
  AUTO_PUSH                             1/0 push branch after commit
  AUTO_PR                               1/0 open or update draft PR after push
  AUTOSAVE_PATCH                        1/0 save git patch snapshots per iteration
  STOP_FILE                             Touch this file to stop the loop

Notes:
  max_iters=0 means run until stopped.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

banner() {
  printf '%s\n' "$1"
}

ensure_session_dir() {
  ensure_private_context_links
  mkdir -p "$WORKTREE/.session"
  mkdir -p "$CODEX_RUNTIME_HOME"
  if [[ -f "$PROJECT_ROOT/.codex/config.toml" ]]; then
    ln -snf "$PROJECT_ROOT/.codex/config.toml" "$CODEX_RUNTIME_HOME/config.toml"
  fi
  if [[ -f "$HOME/.codex/auth.json" && ! -e "$CODEX_RUNTIME_HOME/auth.json" ]]; then
    ln -s "$HOME/.codex/auth.json" "$CODEX_RUNTIME_HOME/auth.json"
  fi
  if [[ -f "$HOME/.codex/version.json" && ! -e "$CODEX_RUNTIME_HOME/version.json" ]]; then
    ln -s "$HOME/.codex/version.json" "$CODEX_RUNTIME_HOME/version.json"
  fi
}

acquire_lock() {
  ensure_session_dir
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" >"$LOCK_PID_FILE"
    return
  fi

  if [[ -f "$LOCK_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
    if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "ERROR: discovery loop already running in $WORKTREE (pid $existing_pid)" >&2
      exit 2
    fi
  fi

  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  printf '%s\n' "$$" >"$LOCK_PID_FILE"
}

release_lock() {
  rm -rf "$LOCK_DIR"
}

ensure_worktree() {
  if [[ -x "$PY" && -f "$MOUNT_GUARD" ]]; then
    "$PY" "$MOUNT_GUARD" --root "$PROJECT_ROOT" >/dev/null 2>&1 || true
  fi

  if [[ -d "$WORKTREE/.git" || -f "$WORKTREE/.git" ]]; then
    return
  fi

  if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    git worktree add "$WORKTREE" "$BRANCH"
    return
  fi

  "$PY" "$PROJECT_ROOT/scripts/tools/worktree_manager.py" create \
    --tool codex \
    --name "$WORKSTREAM" \
    --purpose "$PURPOSE" >/tmp/mnq_autodiscovery_worktree_path.txt

  local created_path
  created_path="$(cat /tmp/mnq_autodiscovery_worktree_path.txt)"
  if [[ "$created_path" != "$WORKTREE" ]]; then
    WORKTREE="$created_path"
  fi
}

write_task_packet() {
  if [[ -f "$TASK_ROUTE" ]]; then
    "$PY" "$TASK_ROUTE" \
      --root "$WORKTREE" \
      --tool codex \
      --task "$TASK_TEXT" \
      --briefing-level read_only >/dev/null || true
  fi
}

refresh_board_stack() {
  (
    cd "$WORKTREE"
    bash "$BOARD_STACK"
  )
}

render_prompt() {
  local prompt_path="$1"
  cat "$PROMPT_TEMPLATE" >"$prompt_path"
  cat <<EOF >>"$prompt_path"

Runtime context:
- Project root: \`$PROJECT_ROOT\`
- Worktree: \`$WORKTREE\`
- Branch: \`$BRANCH\`
- Workstream: \`$WORKSTREAM\`
- Task route packet: \`$WORKTREE/.session/task-route.md\`
- hiROI frontier: \`$FRONTIER_FILE\`
- hiROI capsule: \`$CAPSULE_FILE\`
- Stop file: \`$STOP_FILE\`

Iteration rule:
- Read the frontier and pick from queued candidates first.
- If the queue is honestly exhausted, return \`no_honest_move\` and stop.
- If you edit files, keep the blast radius tight and verify before commit.
- If you make a durable queue decision from real discovery evidence, update \`HANDOFF.md\` in the worktree.
- Do not spend an iteration on \`HANDOFF.md\` housekeeping when the hiROI queue can be advanced or explicitly parked from the capsule and board outputs.
EOF
}

build_frontier() {
  if [[ ! -f "$FRONTIER_TOOL" ]]; then
    return
  fi
  "$PY" "$FRONTIER_TOOL" --root "$WORKTREE" --ledger "$FRONTIER_LEDGER" >"$FRONTIER_FILE"
}

render_capsule() {
  if [[ ! -f "$CAPSULE_TOOL" ]]; then
    return
  fi
  "$PY" "$CAPSULE_TOOL" --root "$WORKTREE" --state "$STATE_FILE" >"$CAPSULE_FILE"
}

validate_result_json() {
  local result_path="$1"
  "$PY" - "$result_path" "$SCHEMA_PATH" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
schema_path = Path(sys.argv[2])
data = json.loads(result_path.read_text(encoding="utf-8"))
schema = json.loads(schema_path.read_text(encoding="utf-8"))

required = schema["required"]
properties = schema["properties"]

missing = [key for key in required if key not in data]
if missing:
    raise SystemExit(f"missing keys: {missing}")

extra = sorted(set(data) - set(properties))
if extra:
    raise SystemExit(f"unexpected keys: {extra}")

for key, spec in properties.items():
    value = data[key]
    expected = spec.get("type")
    if expected == "string" and not isinstance(value, str):
        raise SystemExit(f"{key} must be string")
    if expected == "boolean" and not isinstance(value, bool):
        raise SystemExit(f"{key} must be boolean")
    if expected == "array":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise SystemExit(f"{key} must be array[string]")
    if "enum" in spec and value not in spec["enum"]:
        raise SystemExit(f"{key} must be one of {spec['enum']}")

candidate_id = data["candidate_id"].strip()
frontier_decision = data["frontier_decision"]
status = data["status"]

if frontier_decision == "none" and candidate_id:
    raise SystemExit("candidate_id must be empty when frontier_decision=none")
if frontier_decision != "none" and not candidate_id:
    raise SystemExit("candidate_id required when frontier_decision!=none")
if status in {"parked", "killed", "candidate_advanced", "prereg_written"} and frontier_decision == "none":
    raise SystemExit(f"frontier_decision required for status={status}")
PY
}

json_get() {
  local json_path="$1"
  local key="$2"
  "$PY" - "$json_path" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
value = data.get(key)
if isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, list):
    print("\n".join(str(item) for item in value))
elif value is None:
    print("")
else:
    print(str(value))
PY
}

commit_if_needed() {
  local result_json="$1"
  local should_commit
  should_commit="$(json_get "$result_json" "should_commit")"
  if [[ "$AUTO_COMMIT" != "1" || "$should_commit" != "true" ]]; then
    return
  fi

  if ! git -C "$WORKTREE" diff --quiet || ! git -C "$WORKTREE" diff --cached --quiet || [[ -n "$(git -C "$WORKTREE" ls-files --others --exclude-standard)" ]]; then
    local commit_message
    commit_message="$(json_get "$result_json" "commit_message")"
    if [[ -z "$commit_message" ]]; then
      commit_message="research: advance MNQ autonomous discovery hub"
    fi
    git -C "$WORKTREE" add -A
    git -C "$WORKTREE" commit -m "$commit_message"
  fi
}

push_if_needed() {
  local result_json="$1"
  local should_open_pr
  should_open_pr="$(json_get "$result_json" "should_open_pr")"
  if [[ "$AUTO_PUSH" != "1" ]]; then
    return
  fi

  if [[ "$should_open_pr" == "true" || "$AUTO_PR" == "1" ]]; then
    git -C "$WORKTREE" push -u origin "$BRANCH"
  fi
}

pr_if_needed() {
  local result_json="$1"
  local should_open_pr
  should_open_pr="$(json_get "$result_json" "should_open_pr")"
  if [[ "$AUTO_PR" != "1" || "$should_open_pr" != "true" ]]; then
    return
  fi

  local pr_title pr_body body_file existing_pr
  pr_title="$(json_get "$result_json" "pr_title")"
  pr_body="$(json_get "$result_json" "pr_body")"
  body_file="$(mktemp)"
  printf '%s\n' "$pr_body" >"$body_file"
  existing_pr="$(gh pr list --head "$BRANCH" --json number --jq '.[0].number' 2>/dev/null || true)"

  if [[ -n "$existing_pr" ]]; then
    gh pr edit "$existing_pr" --title "$pr_title" --body-file "$body_file"
  else
    gh pr create --draft --base main --head "$BRANCH" --title "$pr_title" --body-file "$body_file"
  fi
  rm -f "$body_file"
}

snapshot_worktree_state() {
  local iter_dir="$1"
  if [[ "$AUTOSAVE_PATCH" != "1" ]]; then
    return
  fi

  git -C "$WORKTREE" status --short >"$iter_dir/worktree_status.txt"
  git -C "$WORKTREE" diff --binary >"$iter_dir/worktree.diff"
  git -C "$WORKTREE" diff --cached --binary >"$iter_dir/worktree_cached.diff"
  git -C "$WORKTREE" ls-files --others --exclude-standard >"$iter_dir/worktree_untracked.txt"
}

run_initial_exec() {
  local prompt_file="$1"
  local result_file="$2"
  local stdout_file="$3"
  (
    cd "$WORKTREE"
    export CODEX_HOME="$CODEX_RUNTIME_HOME"
    "$CODEX_BIN" exec \
      -p "$PROFILE" \
      --full-auto \
      -o "$result_file" \
      <"$prompt_file" >"$stdout_file" 2>&1
  )
}

run_resume_exec() {
  local prompt_file="$1"
  local result_file="$2"
  local stdout_file="$3"
  (
    cd "$WORKTREE"
    export CODEX_HOME="$CODEX_RUNTIME_HOME"
    "$CODEX_BIN" exec resume \
      --last \
      --full-auto \
      -o "$result_file" \
      - <"$prompt_file" >"$stdout_file" 2>&1
  )
}

ensure_worktree
write_task_packet
ensure_session_dir
acquire_lock
trap release_lock EXIT

if [[ -f "$SESSION_PREFLIGHT" ]]; then
  (
    cd "$WORKTREE"
    "$PY" "$SESSION_PREFLIGHT" --context codex-wsl >/dev/null
  )
fi

banner "=== MNQ AUTONOMOUS DISCOVERY LOOP ==="
banner "Worktree: $WORKTREE"
banner "Branch:   $BRANCH"
banner "Profile:  $PROFILE"
banner "Stop:     $STOP_FILE"
banner "Logs:     $LOG_ROOT"
banner "State:    $STATE_FILE"

iter=1
have_session=0
while :; do
  if [[ -f "$STOP_FILE" ]]; then
    banner "Stop file detected. Exiting."
    break
  fi

  if [[ "$MAX_ITERS" != "0" && "$iter" -gt "$MAX_ITERS" ]]; then
    banner "Reached max iterations ($MAX_ITERS). Exiting."
    break
  fi

  timestamp="$(date +%Y%m%d_%H%M%S)"
  iter_dir="$LOG_ROOT/iter_${iter}_${timestamp}"
  mkdir -p "$iter_dir"
  prompt_file="$iter_dir/prompt.md"
  result_file="$iter_dir/result.json"
  stdout_file="$iter_dir/stdout.log"
  board_log="$iter_dir/board_stack.log"

  banner ""
  banner "--- Iteration $iter ---"
  if ! refresh_board_stack >"$board_log" 2>&1; then
    banner "Board stack refresh failed on iteration $iter. See $board_log"
    break
  fi
  build_frontier
  render_capsule
  cp "$FRONTIER_FILE" "$iter_dir/discovery_frontier.json" 2>/dev/null || true
  cp "$CAPSULE_FILE" "$iter_dir/discovery_capsule.md" 2>/dev/null || true

  render_prompt "$prompt_file"

  if [[ "$have_session" == "1" ]]; then
    if ! run_resume_exec "$prompt_file" "$result_file" "$stdout_file"; then
      banner "codex exec resume failed on iteration $iter. See $stdout_file"
      break
    fi
  else
    if [[ "$RESUME_MODE" != "never" ]]; then
      if run_resume_exec "$prompt_file" "$result_file" "$stdout_file"; then
        have_session=1
      else
        rm -f "$result_file"
      fi
    fi

    if [[ "$have_session" != "1" ]]; then
      if ! run_initial_exec "$prompt_file" "$result_file" "$stdout_file"; then
        banner "codex exec failed on iteration $iter. See $stdout_file"
        break
      fi
      have_session=1
    fi
  fi

  if [[ ! -s "$result_file" ]]; then
    banner "Missing JSON result on iteration $iter. See $stdout_file"
    break
  fi

  if ! validate_result_json "$result_file"; then
    banner "Invalid JSON result on iteration $iter. See $result_file and $stdout_file"
    break
  fi

  banner "Summary: $(json_get "$result_file" "summary")"
  banner "Status:  $(json_get "$result_file" "status")"
  banner "Verify:  $(json_get "$result_file" "verification_summary")"

  snapshot_worktree_state "$iter_dir"

  "$PY" - "$STATE_FILE" "$result_file" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
result = json.loads(result_path.read_text(encoding="utf-8"))

state = {"history": []}
if state_path.exists():
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        state = {"history": []}

history = state.setdefault("history", [])
history.append({
    "summary": result["summary"],
    "status": result["status"],
    "candidate_id": result["candidate_id"],
    "frontier_decision": result["frontier_decision"],
    "next_focus": result["next_focus"],
    "continue_running": result["continue_running"],
})
state["history"] = history[-10:]
state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
PY

  "$PY" - "$FRONTIER_LEDGER" "$result_file" <<'PY'
import json
import sys
from pathlib import Path

ledger_path = Path(sys.argv[1])
result = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
candidate_id = result.get("candidate_id", "").strip()
frontier_decision = result.get("frontier_decision", "none").strip()
if not candidate_id or frontier_decision == "none":
    raise SystemExit(0)

ledger = {}
if ledger_path.exists():
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        ledger = {}
ledger[candidate_id] = {
    "frontier_decision": frontier_decision,
    "summary": result["summary"],
    "status": result["status"],
    "next_focus": result["next_focus"],
}
ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
PY

  commit_if_needed "$result_file"
  push_if_needed "$result_file"
  pr_if_needed "$result_file"

  if "$PY" - "$STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
history = state.get("history", [])
if len(history) < 3:
    raise SystemExit(1)
tail = history[-3:]
status = {row["status"] for row in tail}
focus = {row["next_focus"] for row in tail}
raise SystemExit(0 if len(status) == 1 and len(focus) == 1 else 1)
PY
  then
    banner "Loop repetition detected across the last 3 iterations. Exiting fail-closed."
    break
  fi

  if "$PY" - "$STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

history = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("history", [])
tail = history[-2:]
if len(tail) < 2:
    raise SystemExit(1)
only_engine = all(row.get("status") == "runner_updated" and not row.get("candidate_id") for row in tail)
raise SystemExit(0 if only_engine else 1)
PY
  then
    banner "Loop spent 2 consecutive iterations on engine maintenance without acting on a candidate. Exiting fail-closed."
    break
  fi

  if [[ "$(json_get "$result_file" "continue_running")" != "true" ]]; then
    banner "Model requested stop after iteration $iter."
    break
  fi

  iter=$((iter + 1))
done
