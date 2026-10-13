"""Make the aitcert package importable when pytest is run from any directory."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
