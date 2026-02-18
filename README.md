# llm-agent-quality

Structured quality metrics for LLM Function Calling agents.

Record what happens during each agent invocation (turns, tool calls, token usage, nudge behavior) and assert quality thresholds in your test suite.

## Install

```bash
pip install git+https://github.com/eris-ths/llm-agent-quality.git
```

## Quick Start

```python
from llm_agent_quality import AgentMetrics

metrics = AgentMetrics(model="gemini-2.5-flash")

# Record during agent loop
metrics.record_turn(input_tokens=200, output_tokens=80)
metrics.record_tool_call("write_file", turn=1)
metrics.finalize(turns=2, has_response=True)

# Assert quality in tests
assert metrics.first_tool_turn == 1
assert metrics.turns <= 8
assert not metrics.nudge_fired
```

## Quality Thresholds

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_TOOLS_PER_REQUEST` | 6 | Max tool declarations per request |
| `MAX_TURNS_TO_FIRST_TOOL` | 2 | Max turns before first tool call |
| `MAX_TOTAL_TURNS` | 8 | Max turns to completion |
| `NUDGE_RATE_THRESHOLD` | 0.3 | Max acceptable nudge fire rate |

## Test Patterns

`tests/test_agent_metrics.py` includes reusable quality test patterns:

- **Tool filter effectiveness** — verify filtered tool count stays within budget
- **Nudge rate monitoring** — track nudge fire rate across N runs
- **First tool turn latency** — ensure ReAct prompts trigger immediate tool use
- **Total turns budget** — verify completion within turn limit
- **Error resilience** — confirm metrics survive API failures

## License

MIT
