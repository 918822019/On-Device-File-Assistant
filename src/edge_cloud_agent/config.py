"""Configuration helpers for the edge-cloud hybrid agent."""

from dataclasses import dataclass
import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class EdgeConfig:
    """All knobs for the edge (device/local) generation runtime."""

    source: str = os.getenv("EDGE_MODEL_SOURCE", "hf")
    model_id: str = os.getenv("EDGE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    quantized_model_id: str = os.getenv(
        "EDGE_QUANTIZED_MODEL_ID",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
    )
    local_cache_dir: str = os.getenv("EDGE_LOCAL_DIR", "")
    quantization: str = os.getenv("EDGE_QUANTIZATION", "int4-gp32")
    max_input_tokens: int = _env_int("EDGE_MAX_INPUT_TOKENS", 1400)
    max_new_tokens: int = _env_int("EDGE_MAX_NEW_TOKENS", 220)
    temperature: float = _env_float("EDGE_TEMPERATURE", 0.7)
    top_p: float = _env_float("EDGE_TOP_P", 0.95)
    top_k: int = _env_int("EDGE_TOP_K", 0)
    quant_bits: int = _env_int("EDGE_QUANT_BITS", 4)
    quant_group_size: int = _env_int("EDGE_QUANT_GROUP_SIZE", 32)
    use_sampling: bool = _env_bool("EDGE_USE_SAMPLING", True)
    # 推理设备。auto = 交给 accelerate 决定（Apple Silicon 上会落 MPS）；
    # 也可显式指定 cpu / mps / cuda:0。
    # 注意：transformers 5.x 的 SDPA 在 MPS 上会产出 NaN 与非确定性结果，
    # 本机请设为 cpu，详见 docs/KNOWN_ISSUES.md。
    device: str = os.getenv("EDGE_DEVICE", "auto")
    # 权重精度。auto = 沿用模型 config.json 声明的 dtype（gemma-4-E2B 为 bfloat16）；
    # 也可显式指定 bfloat16 / float16 / float32。
    dtype: str = os.getenv("EDGE_DTYPE", "auto")


@dataclass(frozen=True)
class EmbeddingConfig:
    """All knobs for the edge embedding runtime."""

    source: str = os.getenv("EDGE_EMBEDDING_SOURCE", "modelscope")
    model_id: str = os.getenv("EDGE_EMBEDDING_MODEL_ID", "google/embeddinggemma-300m")
    local_cache_dir: str = os.getenv("EDGE_EMBEDDING_LOCAL_DIR", "")
    max_length: int = _env_int("EDGE_EMBEDDING_MAX_LENGTH", 512)
    batch_size: int = _env_int("EDGE_EMBEDDING_BATCH_SIZE", 8)
    trust_remote_code: bool = _env_bool("EDGE_EMBEDDING_TRUST_REMOTE_CODE", True)
    torch_dtype: str = os.getenv("EDGE_EMBEDDING_TORCH_DTYPE", "float16")
    # 推理设备，取值同 EDGE_DEVICE。transformers 5.x 下 MPS 会导致向量
    # 非确定性（同一输入多次运行得到不同余弦值），本机请设为 cpu。
    device: str = os.getenv("EDGE_EMBEDDING_DEVICE", "auto")


@dataclass(frozen=True)
class CloudConfig:
    """All knobs for the cloud LLM API."""

    enabled: bool = _env_bool("CLOUD_ENABLED", False)
    api_base: str = os.getenv("CLOUD_API_BASE", "http://127.0.0.1:8000/v1")
    api_key: str = os.getenv("CLOUD_API_KEY", "")
    model: str = os.getenv("CLOUD_MODEL_ID", "google/gemma-4-e2b-it")
    timeout_seconds: int = _env_int("CLOUD_TIMEOUT_SECONDS", 20)
    max_new_tokens: int = _env_int("CLOUD_MAX_NEW_TOKENS", 512)
    temperature: float = _env_float("CLOUD_TEMPERATURE", 0.5)


@dataclass(frozen=True)
class RouteConfig:
    """Heuristic routing thresholds."""

    use_tiny_first: bool = _env_bool("ROUTE_USE_TINYLLM", True)
    force_cloud_keywords: tuple[str, ...] = (
        "最新",
        "今天",
        "明天",
        "实时",
        "数据库",
        "联网",
        "查一下",
        "需要最新",
        "stock",
        "行情",
    )
    max_input_chars: int = _env_int("ROUTE_MAX_INPUT_CHARS", 1800)
    min_edge_confidence: float = _env_float("ROUTE_MIN_EDGE_CONFIDENCE", 0.50)


@dataclass(frozen=True)
class ExpenseConfig:
    """Knobs for the reimbursement workflow MVP."""

    store_path: str = os.getenv("EXPENSE_STORE_PATH", "data/expense_store.jsonl")
    default_claim_id: str = os.getenv("EXPENSE_DEFAULT_CLAIM_ID", "default-claim")
    required_doc_types: str = os.getenv(
        "EXPENSE_REQUIRED_DOC_TYPES",
        "invoice,bank_transfer,receipt,approval",
    )
    max_search_limit: int = _env_int("EXPENSE_MAX_SEARCH_LIMIT", 50)
    enable_embedding_search: bool = _env_bool("EXPENSE_ENABLE_EMBEDDING_SEARCH", True)
    watch_dir: str = os.getenv("EXPENSE_WATCH_DIR", "")
    watch_interval_seconds: int = _env_int("EXPENSE_WATCH_INTERVAL_SECONDS", 120)
    watch_file_suffixes: str = os.getenv(
        "EXPENSE_WATCH_FILE_SUFFIXES",
        ".txt,.md,.json,.csv,.log,.pdf,.png,.jpg,.jpeg",
    )
    watch_recursive: bool = _env_bool("EXPENSE_WATCH_RECURSIVE", True)


@dataclass(frozen=True)
class PersonalFileConfig:
    """Knobs for personal file search and indexing."""

    store_path: str = os.getenv("FILE_MEMORY_STORE_PATH", "data/personal_file_store.jsonl")
    source_dir: str = os.getenv("FILE_MEMORY_SOURCE_DIR", "")
    scan_interval_seconds: int = _env_int("FILE_MEMORY_SCAN_INTERVAL_SECONDS", 120)
    scan_file_suffixes: str = os.getenv(
        "FILE_MEMORY_SCAN_FILE_SUFFIXES",
        ".txt,.md,.json,.csv,.log,.pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.png,.jpg,.jpeg,.gif,.bmp,.webp,.mp4,.mov,.m4a,.mp3,.wav",
    )
    scan_recursive: bool = _env_bool("FILE_MEMORY_SCAN_RECURSIVE", True)
    top_k_default: int = _env_int("FILE_MEMORY_TOP_K_DEFAULT", 8)
    max_search_query_len: int = _env_int("FILE_MEMORY_MAX_QUERY_LEN", 120)
    enable_faiss: bool = _env_bool("FILE_MEMORY_ENABLE_FAISS", True)
    faiss_index_path: str = os.getenv(
        "FILE_MEMORY_FAISS_INDEX_PATH",
        "data/personal_file_faiss.index",
    )
    faiss_candidate_multiplier: int = _env_int("FILE_MEMORY_FAISS_CANDIDATE_MULTIPLIER", 4)
    faiss_text_weight: float = _env_float("FILE_MEMORY_FAISS_TEXT_WEIGHT", 0.65)
    faiss_semantic_weight: float = _env_float("FILE_MEMORY_FAISS_SEMANTIC_WEIGHT", 0.30)
    faiss_clue_weight: float = _env_float("FILE_MEMORY_FAISS_CLUE_WEIGHT", 0.25)
    faiss_version_bonus: float = _env_float("FILE_MEMORY_FAISS_VERSION_BONUS", 0.20)
