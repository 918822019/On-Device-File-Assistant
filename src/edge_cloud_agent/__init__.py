"""Entry module for edge-cloud hybrid agent package.

在这里加载 .env 是有意为之：config.py 的所有 Config 都是
`@dataclass(frozen=True)` 且默认值直接写成 `os.getenv(...)`，这些默认值在
**模块 import 时求值一次**（见 docs/KNOWN_ISSUES.md O6）。因此 .env 必须在任何
子模块被 import 之前注入到 os.environ，否则读到的全是代码里的硬编码默认值。

放在包的 __init__.py 里可以保证：无论从 uvicorn、make run、pytest 还是直接
`import edge_cloud_agent` 进入，.env 都会先生效。
"""

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv 是声明依赖，缺失时不应阻断导入
    load_dotenv = None

if load_dotenv is not None:
    # 仓库根目录 = 本文件的上上级（src/edge_cloud_agent/__init__.py -> src -> repo）
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    _ENV_FILE = _REPO_ROOT / ".env"
    if _ENV_FILE.is_file():
        # override=False：调用方显式设置的环境变量优先于 .env 文件，
        # 这样才能用 `CLOUD_ENABLED=true bash scripts/run.sh` 之类的方式临时覆盖。
        load_dotenv(_ENV_FILE, override=False)
