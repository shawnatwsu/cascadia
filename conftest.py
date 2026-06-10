"""Root conftest so `import cascadia` resolves under a bare `pytest` invocation.

pytest inserts the directory of the topmost conftest.py onto sys.path (prepend
import mode). Without this, `pytest tests/` (as CI runs it) only adds tests/ to
the path and fails to import the cascadia package at the repo root — whereas
`python -m pytest` happens to work because it adds the CWD. This file makes both
invocations behave identically.
"""
