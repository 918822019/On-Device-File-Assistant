"""Pytest 全局配置：把 src 加入 sys.path，测试内直接 import edge_cloud_agent。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
