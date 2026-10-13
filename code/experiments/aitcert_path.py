"""Make the aitcert package importable from the experiment scripts."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aitcert")))
from aitcert import stats, model  # noqa: E402,F401
