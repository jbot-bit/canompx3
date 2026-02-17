"""
AI Chat module for the Streamlit dashboard.

Uses Ollama (qwen2.5-coder:7b) with SOP-compliant system prompt.
- Auto-executes read-only SQL via [QUERY:SELECT ...]
- Proposes commands via [SUGGEST_CMD:command] (user clicks to run)
- Loads MarketState via [MARKET_STATE:YYYY-MM-DD]
"""

import re
from datetime import date

import streamlit as st


from ui.db_reader import get_schema_summary, get_table_counts, get_date_ranges, query_df
from ui.sandbox_runner import run_sandboxed, promote_sandbox, discard_sandbox

# Tag patterns for structured AI output
QUERY_PATTERN = re.compile(r"\[QUERY:(.*?)\]", re.DOTALL)
SUGGEST_CMD_PATTERN = re.compile(r"\[SUGGEST_CMD:(.*?)\]", re.DOTALL)
MARKET_STATE_PATTERN = re.compile(r"\[MARKET_STATE:(\d{4}-\d{2}-\d{2})\]")

def _build_system_prompt() -> str:
    """Build the system prompt with SOP rules + schema context."""
    # Load SOP files
    sysop_path = PROJECT_ROOT / "sysop.txt"
    gates_path = PROJECT_ROOT / "sysopgates.txt"

    sysop_text = sysop_path.read_text(encoding="utf-8") if sysop_path.exists() else ""
    gates_text = gates_path.read_text(encoding="utf-8") if gates_path.exists() else ""

    # Get live DB stats
    try:
        counts = get_table_counts()
        ranges = get_date_ranges()
        schema = get_schema_summary()
    except Exception:
        counts = {}
        ranges = {}
        schema = "(DB not available)"

    counts_str = "\n".join(f"  {k}: {v:,}" for k, v in counts.items())
    ranges_str = "\n".join(f"  {k}: {v}" for k, v in ranges.items())

    return f"""You are a Research Analyst for the Canompx3 Gold (MGC) trading project.

=== SOP RULES (MANDATORY) ===
{sysop_text}

=== GATED EXECUTION RULES (MANDATORY) ===
{gates_text}

=== DATABASE SCHEMA ===
{schema}

=== CURRENT DB STATS ===
Row counts:
{counts_str}

Date ranges:
{ranges_str}

=== STRUCTURED OUTPUT TAGS ===
You can use these tags in your responses. The dashboard will parse and execute them:

1. [QUERY:SELECT ...] - Auto-executed read-only SQL. Result shown as table.
   Example: [QUERY:SELECT COUNT(*) AS cnt FROM validated_setups WHERE orb_label='0900']

2. [SUGGEST_CMD:command] - Shown as a button the user clicks to run.
   Example: [SUGGEST_CMD:python -m pytest tests/ -x -q]

3. [MARKET_STATE:YYYY-MM-DD] - Load and display MarketState for that date.
   Example: [MARKET_STATE:2025-06-15]

=== RULES FOR TAGS ===
- ONLY use [QUERY:...] for SELECT/WITH statements. Never INSERT/UPDATE/DELETE.
- ONLY use [SUGGEST_CMD:...] for commands. Never auto-execute tests/discovery/validation.
- Use absolute paths in commands. Project root: {PROJECT_ROOT}
- Sandbox DB: C:\\db\\gold_sandbox.db
- Main DB: {PROJECT_ROOT / 'gold.db'}
"""

def _try_ollama_chat(messages: list[dict]) -> tuple[str | None, bool]:
    """Call Ollama and return (response_text, is_error).

    Returns (None, True) if Ollama package unavailable.
    Returns (error_string, True) if Ollama call fails.
    Returns (response_text, False) on success.
    """
    try:
        import ollama
        response = ollama.chat(
            model="qwen2.5-coder:7b",
            messages=messages,
        )
        return response["message"]["content"], False
    except ImportError:
        return None, True
    except Exception as e:
        return f"Ollama error: {e}", True

def _process_queries(text: str) -> list[tuple[str, object]]:
    """Find and execute [QUERY:...] tags. Returns list of (sql, result_df_or_error)."""
    results = []
    for match in QUERY_PATTERN.finditer(text):
        sql = match.group(1).strip()
        try:
            df = query_df(sql)
            results.append((sql, df))
        except Exception as e:
            results.append((sql, str(e)))
    return results

def _extract_suggestions(text: str) -> list[str]:
    """Extract [SUGGEST_CMD:...] commands."""
    return [m.group(1).strip() for m in SUGGEST_CMD_PATTERN.finditer(text)]

def _extract_market_states(text: str) -> list[str]:
    """Extract [MARKET_STATE:date] dates."""
    return [m.group(1) for m in MARKET_STATE_PATTERN.finditer(text)]

def _clean_tags(text: str) -> str:
    """Remove structured tags from display text (they're rendered separately)."""
    text = QUERY_PATTERN.sub("", text)
    text = SUGGEST_CMD_PATTERN.sub("", text)
    text = MARKET_STATE_PATTERN.sub("", text)
    return text.strip()

def init_chat_state():
    """Initialize chat session state."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = _build_system_prompt()
    if "pending_commands" not in st.session_state:
        st.session_state.pending_commands = []
    if "sandbox_active" not in st.session_state:
        st.session_state.sandbox_active = False

def render_chat():
    """Render the AI chat in the sidebar."""
    init_chat_state()

    st.sidebar.markdown("---")
    st.sidebar.subheader("AI Research Assistant")
    st.sidebar.caption("Local Ollama (qwen2.5-coder:7b) -- SOP-compliant")

    # Display chat history
    chat_container = st.sidebar.container(height=400)
    with chat_container:
        for msg in st.session_state.chat_messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                st.markdown(f"**You:** {content}")
            elif role == "assistant":
                st.markdown(f"**AI:** {content}")
            elif role == "query_result":
                st.caption(f"Query: `{msg.get('sql', '')}`")
                if isinstance(msg.get("result"), str):
                    st.error(msg["result"])
                else:
                    st.dataframe(msg["result"], use_container_width=True)
            elif role == "command_output":
                with st.expander(f"Output: {msg.get('cmd', '')}", expanded=False):
                    st.code(msg.get("output", ""), language="text")
            elif role == "market_state":
                st.info(f"MarketState loaded for {msg.get('date', '')}")
                if msg.get("data"):
                    st.json(msg["data"])

    # Pending command buttons
    if st.session_state.pending_commands:
        st.sidebar.markdown("**Suggested commands:**")
        for i, cmd in enumerate(st.session_state.pending_commands):
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.code(cmd, language="bash")
            with col2:
                if st.button("Run", key=f"run_cmd_{i}"):
                    _execute_suggested_command(cmd)
                    st.rerun()

    # Sandbox management
    if st.session_state.sandbox_active:
        st.sidebar.warning("Sandbox DB active")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Promote to main"):
                if promote_sandbox():
                    st.session_state.sandbox_active = False
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": "Sandbox promoted to main DB.",
                    })
                    st.rerun()
        with col2:
            if st.button("Discard"):
                if discard_sandbox():
                    st.session_state.sandbox_active = False
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": "Sandbox discarded.",
                    })
                    st.rerun()

    # Input
    user_input = st.sidebar.chat_input("Ask about strategies, data, or commands...")
    if user_input:
        _handle_user_message(user_input)
        st.rerun()

def _handle_user_message(user_input: str):
    """Process a user message through Ollama and parse the response."""
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    st.session_state.pending_commands = []

    # Build messages for Ollama
    ollama_messages = [{"role": "system", "content": st.session_state.system_prompt}]
    for msg in st.session_state.chat_messages:
        if msg["role"] in ("user", "assistant"):
            ollama_messages.append({"role": msg["role"], "content": msg["content"]})

    # Get AI response
    response, is_error = _try_ollama_chat(ollama_messages)
    if response is None:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": "Ollama not available. Install with: `pip install ollama` and ensure the server is running.",
        })
        return
    if is_error:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response,
        })
        return

    # Parse structured tags
    query_results = _process_queries(response)
    suggestions = _extract_suggestions(response)
    market_dates = _extract_market_states(response)

    # Add cleaned response
    clean_text = _clean_tags(response)
    if clean_text:
        st.session_state.chat_messages.append({"role": "assistant", "content": clean_text})

    # Add query results
    for sql, result in query_results:
        st.session_state.chat_messages.append({
            "role": "query_result",
            "sql": sql,
            "result": result,
            "content": f"Query result for: {sql[:60]}...",
        })

    # Store suggestions for button rendering
    st.session_state.pending_commands = suggestions

    # Load market states
    for date_str in market_dates:
        _load_market_state(date_str)

def _execute_suggested_command(command: str):
    """Execute a suggested command through the sandbox runner."""
    exit_code, output, was_sandboxed = run_sandboxed(command)

    if was_sandboxed:
        st.session_state.sandbox_active = True

    st.session_state.chat_messages.append({
        "role": "command_output",
        "cmd": command,
        "output": output,
        "exit_code": exit_code,
        "content": f"Ran: {command}",
    })
    st.session_state.pending_commands = []

def _load_market_state(date_str: str):
    """Load MarketState for a date and add to chat."""
    try:
        from trading_app.market_state import MarketState
        ms = MarketState.from_trading_day(
            trading_day=date.fromisoformat(date_str),
            db_path=str(PROJECT_ROOT / "gold.db"),
            orb_minutes=5,
        )
        # Convert to dict for display
        orbs_data = {}
        for label, orb in ms.orbs.items():
            orbs_data[label] = {
                "high": orb.high,
                "low": orb.low,
                "size": orb.size,
                "break_dir": orb.break_dir,
                "outcome": orb.outcome,
            }

        data = {
            "trading_day": date_str,
            "rsi_14": ms.rsi_14,
            "orbs": orbs_data,
            "signals": {
                "reversal_active": ms.signals.reversal_active,
                "chop_detected": ms.signals.chop_detected,
                "continuation": ms.signals.continuation,
            },
        }
        st.session_state.chat_messages.append({
            "role": "market_state",
            "date": date_str,
            "data": data,
            "content": f"MarketState for {date_str}",
        })
    except Exception as e:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"Failed to load MarketState for {date_str}: {e}",
        })
