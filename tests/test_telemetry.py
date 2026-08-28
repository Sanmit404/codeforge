from codeforge.telemetry import record, summarize


def test_events_are_logged_and_rolled_up(tmp_path, monkeypatch):
    log = tmp_path / "run_log.jsonl"
    monkeypatch.setenv("RUN_LOG", str(log))

    record("tool_call", tool="search_repository", duration_ms=100, success=True)
    record("tool_call", tool="push_files", duration_ms=300, success=False)
    record("retrieval", matches=4, mode="hybrid")
    record("plan_rejected", unknown=["billing.py"])
    record("validation", passed=True, attempt=0)

    summary = summarize(log)

    assert summary["tool_calls"] == 2
    assert summary["tool_failures"] == 1
    assert summary["mean_tool_ms"] == 200.0
    assert summary["retrieval_queries"] == 1
    assert summary["rejected_plans"] == 1
    assert summary["validation_passes"] == 1
