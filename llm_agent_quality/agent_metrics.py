"""Agent quality metrics — LLM Function Calling エージェントの品質計測.

LLM エージェントの1回の呼び出し（ask/turn）で何が起きたかを構造化して記録する。
テストの延長として品質基準を assertion し、実API評価のデータソースにもなる。

Usage:
    metrics = AgentMetrics(model="gemini-2.5-flash")
    # ... ターンループ内で記録 ...
    metrics.record_turn(input_tokens=200, output_tokens=80)
    metrics.record_tool_call("write_file", turn=1)
    metrics.record_nudge()
    metrics.finalize(turns=3, has_response=True)

    # テストで品質基準をアサーション
    assert metrics.tool_calls <= 5
    assert metrics.first_tool_turn == 1
    assert not metrics.nudge_fired

Quality Thresholds (参考値):
    MAX_TOOLS_PER_REQUEST = 6       # リクエストあたりのツール定義数上限
    MAX_TURNS_TO_FIRST_TOOL = 2     # 最初のツール呼び出しまでの最大ターン数
    MAX_TOTAL_TURNS = 8             # 完了までの最大ターン数
    NUDGE_RATE_THRESHOLD = 0.3      # ナッジ発火率の許容上限
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentMetrics:
    """1回の LLM エージェント呼び出しのパフォーマンスメトリクス.

    Attributes:
        turns: 完了までのターン数
        tool_calls: ツール呼び出し回数
        nudge_fired: テキストのみ応答後のナッジ（再促進）が発火したか
        input_tokens: 入力トークン数（全ターン累計）
        output_tokens: 出力トークン数（全ターン累計）
        tool_declarations_count: リクエストに含まれたツール定義数
        first_tool_turn: 最初のツール呼び出しが出たターン番号（None=ツール未使用）
        tool_names: 呼び出されたツール名のリスト（順序保持）
        model: 使用モデル名
        empty_response: 最終応答が空だったか
    """

    turns: int = 0
    tool_calls: int = 0
    nudge_fired: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    tool_declarations_count: int = 0
    first_tool_turn: int | None = None
    tool_names: list[str] = field(default_factory=list)
    model: str = ""
    empty_response: bool = False

    def record_turn(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """ターンごとのトークン使用量を累積."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def record_tool_call(self, tool_name: str, turn: int) -> None:
        """ツール呼び出しを記録."""
        self.tool_calls += 1
        self.tool_names.append(tool_name)
        if self.first_tool_turn is None:
            self.first_tool_turn = turn

    def record_nudge(self) -> None:
        """ナッジ（再促進）の発火を記録."""
        self.nudge_fired = True

    def finalize(self, turns: int, has_response: bool) -> None:
        """メトリクスを確定."""
        self.turns = turns
        self.empty_response = not has_response

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def unique_tools_used(self) -> set[str]:
        return set(self.tool_names)


# ─── 品質しきい値（テスト・評価で import して使う） ───

MAX_TOOLS_PER_REQUEST = 6
MAX_TURNS_TO_FIRST_TOOL = 2
MAX_TOTAL_TURNS = 8
NUDGE_RATE_THRESHOLD = 0.3
