from codeforge.graph import Workflow
from codeforge.state import State


def test_a_plan_grounded_in_retrieval_is_accepted():
    state = State(retrieved_paths=["auth.py", "tests/test_auth.py"])

    rejection = Workflow._reject_ungrounded_plan(
        state, {"files_to_touch": ["auth.py"]}, "call-1"
    )

    assert rejection is None


def test_a_plan_naming_an_unseen_file_is_sent_back():
    state = State(retrieved_paths=["auth.py"])

    rejection = Workflow._reject_ungrounded_plan(
        state, {"files_to_touch": ["auth.py", "billing/invoice.py"]}, "call-1"
    )

    assert rejection is not None
    assert rejection.goto == "planner"
    assert "billing/invoice.py" in rejection.update["messages"][0].content


def test_passing_validation_is_taken_from_the_tool_output():
    state = State(current_agent="validator")
    update = {"messages": []}

    goto = Workflow._apply_validation(state, {"passed": True}, update)

    assert goto == "validator"
    assert update["validation_status"] == "passed"


def test_a_failure_costs_one_repair_attempt():
    state = State(current_agent="validator", max_validation_attempts=2)
    update = {"messages": []}

    goto = Workflow._apply_validation(state, {"passed": False}, update)

    assert goto == "validator"
    assert update["validation_attempts"] == 1
    assert update["validation_status"] == "failed"


def test_the_last_attempt_hands_the_run_back_to_a_human():
    state = State(current_agent="validator", validation_attempts=1, max_validation_attempts=2)
    update = {"messages": []}

    goto = Workflow._apply_validation(state, {"passed": False}, update)

    assert goto == "orchestrator"
    assert update["messages"]
