"""Shared test setup, and the one convention this suite follows.

Every guard's docstring names the mutation that makes it red -- the placebo. A test whose placebo
cannot be named is not testing anything (convention taken from r-ms/mini-jev `tests/test_harness.py`).
"""
import os

# The dev box exports a SOCKS proxy that httpx/requests cannot use without
# socksio, and that localhost calls must not go through anyway.
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_var, None)
