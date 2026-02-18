"""AgentMetrics の単体テスト + 品質基準リファレンス.

このファイルは2つの役割を持つ:
1. AgentMetrics dataclass の動作検証
2. LLM エージェントの品質基準テストパターンのリファレンス実装

品質基準パターン（他プロジェクトで流用可能）:
  - ツールフィルタリングの効果検証
  - ナッジ発火率の監視
  - マルチターンでのメトリクス精度
  - エラー耐性
"""

import pytest
from dataclasses import asdict

from llm_agent_quality import (
    AgentMetrics,
    MAX_TOOLS_PER_REQUEST,
    MAX_TURNS_TO_FIRST_TOOL,
    MAX_TOTAL_TURNS,
    NUDGE_RATE_THRESHOLD,
)


class TestAgentMetricsBasic:
    """AgentMetrics の基本動作"""

    def test_default_values(self):
        m = AgentMetrics()
        assert m.turns == 0
        assert m.tool_calls == 0
        assert m.nudge_fired is False
        assert m.input_tokens == 0
        assert m.output_tokens == 0
        assert m.tool_declarations_count == 0
        assert m.first_tool_turn is None
        assert m.tool_names == []
        assert m.model == ""
        assert m.empty_response is False

    def test_serializable_to_dict(self):
        m = AgentMetrics(turns=3, tool_calls=2, model="gemini-2.5-flash")
        d = asdict(m)
        assert isinstance(d, dict)
        assert d["turns"] == 3
        assert d["model"] == "gemini-2.5-flash"

    def test_tool_names_instance_independence(self):
        """tool_names のデフォルトがインスタンス間で共有されない"""
        m1 = AgentMetrics()
        m2 = AgentMetrics()
        m1.tool_names.append("write_file")
        assert m2.tool_names == []


class TestRecordMethods:
    """record_* メソッドの累積動作"""

    def test_record_turn_accumulates_tokens(self):
        m = AgentMetrics()
        m.record_turn(input_tokens=100, output_tokens=50)
        m.record_turn(input_tokens=200, output_tokens=80)
        assert m.input_tokens == 300
        assert m.output_tokens == 130
        assert m.total_tokens == 430

    def test_record_tool_call(self):
        m = AgentMetrics()
        m.record_tool_call("write_file", turn=1)
        m.record_tool_call("read_file", turn=2)
        m.record_tool_call("write_file", turn=2)
        assert m.tool_calls == 3
        assert m.first_tool_turn == 1
        assert m.tool_names == ["write_file", "read_file", "write_file"]
        assert m.unique_tools_used == {"write_file", "read_file"}

    def test_first_tool_turn_only_set_once(self):
        m = AgentMetrics()
        m.record_tool_call("a", turn=3)
        m.record_tool_call("b", turn=5)
        assert m.first_tool_turn == 3  # 最初の値を保持

    def test_record_nudge(self):
        m = AgentMetrics()
        assert m.nudge_fired is False
        m.record_nudge()
        assert m.nudge_fired is True

    def test_finalize(self):
        m = AgentMetrics()
        m.finalize(turns=5, has_response=True)
        assert m.turns == 5
        assert m.empty_response is False

    def test_finalize_empty_response(self):
        m = AgentMetrics()
        m.finalize(turns=1, has_response=False)
        assert m.empty_response is True


class TestQualityThresholds:
    """品質しきい値の妥当性検証"""

    def test_thresholds_reasonable(self):
        assert 1 <= MAX_TOOLS_PER_REQUEST <= 10
        assert 1 <= MAX_TURNS_TO_FIRST_TOOL <= 5
        assert 1 <= MAX_TOTAL_TURNS <= 15
        assert 0 < NUDGE_RATE_THRESHOLD < 1.0


# ─── 品質基準テストパターン（リファレンス実装） ───


class TestQualityPatterns:
    """品質基準テストのパターン集.

    他プロジェクトでの流用を意図した、LLM エージェント品質検証の書き方リファレンス。
    実APIテストでは、これらのパターンをベースに assert を追加する。
    """

    def test_pattern_tool_filter_effectiveness(self):
        """パターン: ツールフィルタの効果を定量検証.

        allowed_tools で絞った結果が MAX_TOOLS_PER_REQUEST 以下であることを確認。
        実装では build_tools(allowed_tools=...) の戻り値で検証する。
        """
        # シミュレーション: create story は4ツール
        allowed = ["write_file", "create_directory", "append_file", "read_file"]
        assert len(allowed) <= MAX_TOOLS_PER_REQUEST

    def test_pattern_nudge_rate_monitoring(self):
        """パターン: ナッジ発火率の監視.

        N回実行して、ナッジが発火した割合が NUDGE_RATE_THRESHOLD 以下かを検証。
        実APIテストではループ内で metrics.nudge_fired をカウントする。
        """
        # シミュレーション: 10回中2回ナッジ発火
        results = [AgentMetrics(nudge_fired=False) for _ in range(8)]
        results += [AgentMetrics(nudge_fired=True) for _ in range(2)]
        nudge_rate = sum(1 for m in results if m.nudge_fired) / len(results)
        assert nudge_rate <= NUDGE_RATE_THRESHOLD

    def test_pattern_first_tool_turn_latency(self):
        """パターン: 最初のツール呼び出しまでのレイテンシ検証.

        ReAct プロンプトが効いていれば、first_tool_turn ≤ MAX_TURNS_TO_FIRST_TOOL。
        """
        m = AgentMetrics()
        m.record_tool_call("write_file", turn=1)
        m.finalize(turns=2, has_response=True)
        assert m.first_tool_turn is not None
        assert m.first_tool_turn <= MAX_TURNS_TO_FIRST_TOOL

    def test_pattern_total_turns_budget(self):
        """パターン: ターン数の予算内完了.

        create 系タスクが MAX_TOTAL_TURNS 以内に完了するかを検証。
        """
        m = AgentMetrics()
        for i in range(1, 4):
            m.record_tool_call("write_file", turn=i)
            m.record_turn(input_tokens=200, output_tokens=100)
        m.finalize(turns=3, has_response=True)
        assert m.turns <= MAX_TOTAL_TURNS

    def test_pattern_error_resilience(self):
        """パターン: エラー時もメトリクスが記録される.

        API例外やツール実行エラーが起きても、finalize() でメトリクスが残ることを確認。
        """
        m = AgentMetrics(model="gemini-test")
        m.record_turn(input_tokens=100, output_tokens=0)
        # エラーが起きた（has_response=False）
        m.finalize(turns=1, has_response=False)
        assert m.turns == 1
        assert m.empty_response is True
        assert m.model == "gemini-test"  # モデル情報は保持
