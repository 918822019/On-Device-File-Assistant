"""锁定 config.py 默认值与 .env.example 的一致性。

`.env` 被 gitignore，所以**新克隆环境跑的就是 config.py 的默认值**。这两处一旦
漂移，代码里的默认值就会和文档宣称的行为相反，而且不会有任何报错 —— 只是静默
地走进已知有问题的路径。历史上曾同时漂移 6 项：

    EDGE_MODEL_ID                TinyLlama-1.1B   vs  google/gemma-4-E2B-it
    EDGE_QUANTIZED_MODEL_ID      TheBloke GPTQ    vs  空（与 model_id 指向不同模型，
                                                      打开量化会静默加载无关权重）
    EDGE_QUANTIZATION            int4-gp32        vs  none（GPTQ/BNB 链路已弃用且需 CUDA）
    EDGE_DEVICE                  auto             vs  cpu（auto 在 Apple Silicon 落 MPS，
                                                      transformers 5.x 下产出 NaN）
    EDGE_EMBEDDING_TORCH_DTYPE   float16          vs  bfloat16（MPS+float16 输出全 NaN 向量）
    EDGE_EMBEDDING_DEVICE        auto             vs  cpu（MPS 向量非确定性会污染语义检索）

项目未安装任何 linter，这类漂移没有别的手段能发现，故用测试锁住。

默认值用 AST 静态解析而非 import config：包 __init__ 会 load_dotenv()，import 拿到
的是「被本机 .env 覆盖后」的值，恰好掩盖了要测的东西。
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PY = REPO_ROOT / "src" / "edge_cloud_agent" / "config.py"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

_ENV_HELPERS = {"getenv", "_env_int", "_env_float", "_env_bool"}

# 刻意允许的不一致：key -> 原因。目前为空 —— 任何分歧都应当被修掉或在此显式登记。
ALLOWED_DIVERGENCE: dict[str, str] = {}


def _parse_config_defaults() -> dict[str, object]:
    """静态提取 config.py 里 os.getenv(KEY, default) / _env_*(KEY, default) 的字面默认值。"""

    tree = ast.parse(CONFIG_PY.read_text(encoding="utf-8"))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in _ENV_HELPERS or len(node.args) < 2:
            continue
        key = node.args[0]
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        try:
            defaults[key.value] = ast.literal_eval(node.args[1])
        except ValueError:  # 非字面量默认值（如函数调用）不在本测试射程内
            continue
    return defaults


def _parse_env_example() -> dict[str, str]:
    """解析 .env.example 的 KEY=VALUE，跳过注释与空行。"""

    values: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def _equivalent(default: object, example: str) -> bool:
    """比较默认值与模板值，容忍纯格式差异（0.3 vs 0.30、True vs true）。"""

    if isinstance(default, bool):
        return example.lower() in {"true", "1", "yes"} if default else example.lower() in {
            "",
            "false",
            "0",
            "no",
        }
    if isinstance(default, (int, float)):
        try:
            return float(default) == float(example)
        except ValueError:
            return False
    return str(default) == example


def test_config_defaults_match_env_example():
    """config.py 的每个默认值都必须与 .env.example 一致（或登记在 ALLOWED_DIVERGENCE）。"""

    defaults = _parse_config_defaults()
    example = _parse_env_example()
    assert defaults, "未能从 config.py 解析出任何默认值，解析逻辑可能已失效"

    drifted: list[str] = []
    undocumented: list[str] = []
    for key, default in sorted(defaults.items()):
        if key not in example:
            undocumented.append(f"{key}（config 默认 {default!r}，但 .env.example 未列出）")
            continue
        if key in ALLOWED_DIVERGENCE:
            continue
        if not _equivalent(default, example[key]):
            drifted.append(f"{key}: config 默认 {default!r} != .env.example {example[key]!r}")

    assert not drifted, (
        "config.py 默认值与 .env.example 漂移。无 .env 的新环境会跑默认值，"
        "漂移意味着行为与文档相反且无任何报错：\n  " + "\n  ".join(drifted)
    )
    assert not undocumented, (
        "以下配置项未出现在 .env.example，使用者无从得知其存在：\n  " + "\n  ".join(undocumented)
    )


def test_env_example_keys_are_all_consumed():
    """.env.example 里每个键都必须真的被代码读取，避免模板留下已失效的开关。

    部分键由 config.py 之外的模块读取（APP_LOG_LEVEL 在 logging_config.py、
    FILE_MEMORY_WSL_PATH_MAP / FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP 在
    common/path_utils.py），因此扫描整个 src/ 与 scripts/ 而非只看 config.py。
    """

    example = _parse_env_example()
    assert example, ".env.example 解析为空"

    haystack_parts: list[str] = []
    for base in ("src", "scripts"):
        for path in (REPO_ROOT / base).rglob("*"):
            if path.suffix in {".py", ".sh", ".bat", ".ps1"} and "__pycache__" not in path.parts:
                haystack_parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    haystack = "\n".join(haystack_parts)
    assert haystack.strip(), "未扫描到任何源码文件"

    orphans = [key for key in sorted(example) if key not in haystack]
    assert not orphans, f".env.example 中的这些键没有任何代码读取：{orphans}"


@pytest.mark.parametrize("key", sorted(ALLOWED_DIVERGENCE) or ["<empty>"])
def test_allowed_divergence_still_exists(key):
    """ALLOWED_DIVERGENCE 里的条目应当定期复查：分歧消失后就该把豁免删掉。"""

    if key == "<empty>":
        pytest.skip("当前无豁免项")
    defaults = _parse_config_defaults()
    example = _parse_env_example()
    assert key in defaults and key in example, f"豁免项 {key} 已不存在，请清理 ALLOWED_DIVERGENCE"
    assert not _equivalent(defaults[key], example[key]), (
        f"{key} 已恢复一致（{ALLOWED_DIVERGENCE[key]}），请从 ALLOWED_DIVERGENCE 移除该豁免"
    )
