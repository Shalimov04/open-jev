"""Shared test setup, and the one convention this suite follows.

Every guard's docstring names the mutation that makes it red -- the placebo. A test whose placebo
cannot be named is not testing anything (convention taken from r-ms/mini-jev `tests/test_harness.py`).
"""
import os

# The dev box exports a SOCKS proxy that httpx/requests cannot use without
# socksio, and that localhost calls must not go through anyway.
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_var, None)

# With the proxies stripped above, a direct HF request on this box does not fail, it hangs --
# three test runs sat for up to six hours on one HEAD call. Everything the suite needs is cached,
# so go offline when the cache can serve it; CI has no cache and real network, and still downloads.
_HUB = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.isdir(os.path.join(_HUB, "models--jhu-clsp--mmBERT-small")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
