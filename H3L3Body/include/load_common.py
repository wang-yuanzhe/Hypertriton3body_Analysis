# tools/load_common.py
import sys
import os

def add_common_path():
    common_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'commonTools'))
    if common_path not in sys.path:
        sys.path.insert(0, common_path)
