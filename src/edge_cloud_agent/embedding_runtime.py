"""Edge runtime for local embedding inference."""

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer

try:
    from modelscope import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None

from .config import EmbeddingConfig


@dataclass(frozen=True)
class EmbeddingInferenceResult:
    embeddings: list[list[float]]
    used_model: str


class EdgeEmbeddingRuntime:
    """Embedding runtime for endpoint-side vector generation."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("edge_embedding")
        self.model = None
        self.tokenizer = None
        self.model_id = self._resolve_model_id(cfg)
        self._ready = False
        self._load_model()

    @property
    def ready(self) -> bool:
        return self._ready

    def _resolve_model_id(self, cfg: EmbeddingConfig) -> str:
        if cfg.local_cache_dir:
            return cfg.local_cache_dir
        return cfg.model_id

    def _download_if_modelscope(self, model_id: str) -> str:
        if self.cfg.source.lower() != "modelscope":
            return model_id

        if snapshot_download is None:
            raise RuntimeError("modelscope package not installed. Please run: pip install modelscope")

        cache_dir = self.cfg.local_cache_dir or "/tmp/modelscope_embedding_cache"
        return snapshot_download(model_id, cache_dir=cache_dir)

    def _dtype(self):
        if (self.cfg.torch_dtype or "").lower() in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if (self.cfg.torch_dtype or "").lower() in {"fp16", "float16"}:
            return torch.float16
        if (self.cfg.torch_dtype or "").lower() in {"fp32", "float32"}:
            return torch.float32
        # auto：交给 transformers 读取模型 config.json 的 dtype 字段
        return "auto"

    def _load_model(self) -> None:
        resolved_model = self.model_id

        if self.cfg.source.lower() == "modelscope":
            try:
                resolved_model = self._download_if_modelscope(self.model_id)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("ModelScope 下载失败，回退直接加载模型: %s", exc)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_model,
                trust_remote_code=self.cfg.trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModel.from_pretrained(
                resolved_model,
                dtype=self._dtype(),
                device_map=(self.cfg.device or "auto").strip() or "auto",
                trust_remote_code=self.cfg.trust_remote_code,
            ).eval()
            self._ready = True
            self.logger.info("Embedding model loaded: %s", self.model_id)
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Embedding model load failed: %s", exc)
            self._ready = False
            raise

    @torch.no_grad()
    def embed(self, texts: list[str], normalize: bool = True) -> EmbeddingInferenceResult:
        if not self.ready:
            raise RuntimeError("Embedding model not ready")
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Embedding tokenizer/model missing")

        if not texts:
            return EmbeddingInferenceResult(embeddings=[], used_model=self.model_id)

        if isinstance(texts, tuple):
            texts = list(texts)

        all_embeddings: list[list[float]] = []
        device = self.model.device

        for i in range(0, len(texts), self.cfg.batch_size):
            chunk = texts[i : i + self.cfg.batch_size]
            inputs = self.tokenizer(
                chunk,
                max_length=self.cfg.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            masked = token_embeddings * attention_mask
            lengths = attention_mask.sum(dim=1).clamp(min=1.0)
            pooled = masked.sum(dim=1) / lengths

            if normalize:
                norm = torch.norm(pooled, dim=1, keepdim=True)
                pooled = pooled / norm.clamp(min=1e-12)

            all_embeddings.extend(pooled.detach().cpu().tolist())

        return EmbeddingInferenceResult(
            embeddings=all_embeddings,
            used_model=self.model_id,
        )
