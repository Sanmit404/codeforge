import pytest

from codeforge.routing import handoff, submit_plan

PLAN = {
    "summary": "add refresh tokens",
    "files_to_touch": ["auth.py"],
    "steps": ["add a store", "wire it up"],
    "test_plan": "unit test the store",
}


def _call(arguments, name, call_id="call-1"):
    return {"args": arguments, "name": name, "type": "tool_call", "id": call_id}


@pytest.mark.parametrize(
    "target, destination",
    [
        ("planner", "planner"),
        ("coder", "coder"),
        ("validator", "validator"),
        ("orchestrator", "orchestrator"),
        ("done", "__end__"),
    ],
)
async def test_handoff_reaches_every_destination(target, destination):
    command = await handoff.ainvoke(_call({"target": target, "reason": "next step"}, "handoff"))

    assert command.goto == destination


async def test_handoff_rejects_an_unknown_target():
    with pytest.raises(ValueError):
        await handoff.ainvoke(_call({"target": "designer", "reason": "why not"}, "handoff"))


async def test_submit_plan_sends_the_plan_to_the_coder():
    command = await submit_plan.ainvoke(_call(PLAN, "submit_plan"))

    assert command.goto == "coder"
    assert command.update["plan"]["files_to_touch"] == ["auth.py"]
    assert command.update["current_agent"] == "coder"
