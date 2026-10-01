import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


TRAINER = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/trainer/ppo/ray_trainer.py"


def run_checkpoint_stage(*, elapsed, limit=36000, step=25, save_freq=20, best_only=False, fail_save=False, save_duration=0):
    tree = ast.parse(TRAINER.read_text())
    fit = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "fit")
    stage = next(node.body for node in ast.walk(fit) if isinstance(node, ast.With) and any(
        isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "should_save" for t in stmt.targets)
        for stmt in node.body
    ))
    start = next(i for i, stmt in enumerate(stage) if isinstance(stmt, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "should_save" for t in stmt.targets))
    save = next(i for i in range(start, len(stage)) if isinstance(stage[i], ast.If)
                and isinstance(stage[i].test, ast.Name) and stage[i].test.id == "should_save")
    stop = next(node for node in ast.walk(fit) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == "time_limit_reached")
    increment = next(node for node in ast.walk(fit) if isinstance(node, ast.AugAssign)
                     and isinstance(node.target, ast.Attribute) and node.target.attr == "global_steps"
                     and node.lineno > stage[save].lineno)
    events = []
    runner = SimpleNamespace(global_steps=step, config=SimpleNamespace(trainer=SimpleNamespace(save_freq=save_freq)))

    def save_checkpoint():
        nonlocal elapsed
        events.append(("save", runner.global_steps))
        if fail_save:
            raise RuntimeError("checkpoint write failed")
        elapsed += save_duration

    runner._save_checkpoint = save_checkpoint
    namespace = {
        "self": runner, "time": SimpleNamespace(time=lambda: elapsed), "run_start_time": 0,
        "stop_after_seconds": limit, "elapsed_seconds": elapsed, "save_best_only": best_only,
        "track_best_checkpoint": False, "is_best_val_step": False, "is_last_step": False,
        "marked_timer": lambda *args, **kwargs: nullcontext(), "timing_raw": {},
        "pprint": lambda message: events.append(("message", message)),
        "progress_bar": SimpleNamespace(close=lambda: events.append(("close", runner.global_steps))),
    }
    function = ast.parse("def run():\n    pass\n").body[0]
    function.body = stage[start:save + 1] + [increment, stop]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(TRAINER), "exec"), namespace)
    namespace["run"]()
    return events


@pytest.mark.parametrize("step,best_only,save_freq", [(23, False, 10), (27, False, 10), (25, False, 20), (35, False, 20), (25, True, 20), (25, False, 0)])
def test_deadline_saves_current_step_before_exit(step, best_only, save_freq):
    events = run_checkpoint_stage(elapsed=36605, step=step, best_only=best_only, save_freq=save_freq)
    assert events[0] == ("save", step)
    assert events[-1] == ("close", step + 1)
    assert f"Stopping at step {step}:" in events[1][1]
    assert "checkpoint saved: True" in events[1][1]


def test_regular_checkpoint_is_saved_once_when_deadline_also_expires():
    events = run_checkpoint_stage(elapsed=36000, step=40)
    assert [event for event in events if event[0] == "save"] == [("save", 40)]


@pytest.mark.parametrize("elapsed,limit", [(35999, 36000), (36605, 0)])
def test_no_early_exit_or_extra_save(elapsed, limit):
    assert run_checkpoint_stage(elapsed=elapsed, limit=limit) == []


def test_checkpoint_failure_propagates_instead_of_reporting_success():
    with pytest.raises(RuntimeError, match="checkpoint write failed"):
        run_checkpoint_stage(elapsed=36605, fail_save=True)


def test_deadline_during_checkpoint_write_stops_without_another_step():
    events = run_checkpoint_stage(elapsed=35999, step=30, save_freq=10, save_duration=2)
    assert events[0] == ("save", 30)
    assert events[-1] == ("close", 31)
    assert "checkpoint saved: True" in events[1][1]


def test_disabled_deadline_does_not_stop_after_checkpoint_write():
    assert run_checkpoint_stage(elapsed=35999, limit=0, step=30, save_freq=10, save_duration=2) == [("save", 30)]
