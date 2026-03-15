### Pre-PR Checklist
- [x] This PR is a bug fix, not a new feature or enhancement.

### Summary

`@record_model` called `atexit.register(_auto_save)` inside `init_wrapper` unconditionally, so every instantiation of a decorated model class accumulated a redundant atexit handler. In typical usage (parameter sweeps, repeated runs, or tests), this caused multiple auto-save handlers to fire at program exit, resulting in redundant file writes and, combined with #201, duplicate `simulation_end` events.

### Bug / Issue

Fixes #202

### Root Cause

The `_auto_save` closure and its `atexit.register()` call lived inside `init_wrapper()`, which runs on every `__init__()`. There was no guard to prevent re-registration.

```python
# Before — registers a new handler on EVERY __init__ call
def init_wrapper(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.recorder = SimulationRecorder(model=self, **recorder_kwargs)
    _attach_recorder_to_agents(self, self.recorder)

    def _auto_save():
        ...
    atexit.register(_auto_save)  # ← called every time
```

### Implementation

Added a `_atexit_registered` instance flag that guards the registration. Each model instance registers **at most one** atexit handler:

```python
# After — guarded, registers at most once per instance
if not getattr(self, "_atexit_registered", False):
    def _auto_save():
        ...
    atexit.register(_auto_save)
    self._atexit_registered = True
```

The flag is set on the **instance** (not the class), so:
- Each distinct model instance still gets exactly one handler ✅
- Multiple distinct instances each get their own handler ✅
- Re-initializing the same object won't re-register ✅

### Testing

All existing tests pass, plus 3 new regression tests added:

- `test_multiple_instances_each_get_one_atexit_handler` — 3 instances → exactly 3 handlers (one per instance)
- `test_atexit_not_reregistered_on_reinit` — re-calling `__init__()` on same instance → still 1 handler
- `test_atexit_registered_flag_set_on_instance` — verifies the guard flag is set

```bash
pytest tests/test_recording/test_record_model.py -v  # 15/15 passed
pytest --cov=mesa_llm tests/                         # 278 passed, 91% coverage
pre-commit run --all-files                           # All hooks pass
```
