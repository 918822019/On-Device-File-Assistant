"""Edge runtime for local tiny LLM inference."""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from modelscope import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None

from .config import EdgeConfig


class EdgeInferenceResult:
    text: str
    confidence: float
    used_model: str

    def __init__(self, text: str, confidence: float, used_model: str) -> None:
        self.text = text
        self.confidence = confidence
        self.used_model = used_model


class EdgeRuntime:
    """TinyLLM edge runtime with quantization fallback chain."""

    _QUANT_DIRECT_MODES = {"int4", "int4-gp32", "gp32", "gptq", "gptq-int4"}
    _GPTQ_MODES = {"gptq", "int4-gp32", "gp32", "gptq-int4"}

    def __init__(self, cfg: EdgeConfig) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("edge_runtime")
        self.model = None
        self.tokenizer = None

        self.model_id = self._resolve_model_id(cfg)
        self._ready = False
        self._load_model()

    @property
    def ready(self) -> bool:
        return self._ready

    def _resolve_model_id(self, cfg: EdgeConfig) -> str:
        # Local cache takes highest priority.
        if cfg.local_cache_dir:
            return cfg.local_cache_dir

        if self._is_quant_mode(cfg.quantization) and cfg.quantized_model_id:
            return cfg.quantized_model_id

        return cfg.model_id

    @staticmethod
    def _is_quant_mode(mode: str) -> bool:
        return (mode or "").strip().lower() in EdgeRuntime._QUANT_DIRECT_MODES

    def _download_if_modelscope(self, model_id: str) -> str:
        if self.cfg.source.lower() != "modelscope":
            return model_id

        if snapshot_download is None:
            raise RuntimeError("modelscope package not installed. Please run: pip install modelscope")

        cache_dir = self.cfg.local_cache_dir or "/tmp/modelscope_cache"
        return snapshot_download(model_id, cache_dir=cache_dir)

    def _dtype(self):
        """把 EDGE_DTYPE 解析为 transformers 的 dtype 参数。

        auto 原样透传，由 transformers 读取模型 config.json 的 dtype 字段
        （gemma-4-E2B 为 bfloat16）；其余取值映射到具体 torch dtype。
        """
        spec = (self.cfg.dtype or "auto").strip().lower()
        if spec in {"", "auto"}:
            return "auto"
        if spec in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if spec in {"fp16", "float16", "half"}:
            return torch.float16
        if spec in {"fp32", "float32", "single"}:
            return torch.float32
        self.logger.warning("未知 EDGE_DTYPE=%r，回退 auto", self.cfg.dtype)
        return "auto"

    def _load_model(self) -> None:
        resolved_model = self.model_id

        if self.cfg.source.lower() == "modelscope":
            try:
                resolved_model = self._download_if_modelscope(self.model_id)
            except Exception as exc:
                self.logger.warning("ModelScope 下载失败，回退到直接加载模型: %s", exc)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_model,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = self._load_model_with_quantization(resolved_model)
            self._ready = True
            self.logger.info("Edge model loaded: %s", self.model_id)
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Edge model load failed: %s", exc)
            self._ready = False
            raise

    def _load_model_with_quantization(self, model_source: str):
        mode = (self.cfg.quantization or "").strip().lower()
        device_map = (self.cfg.device or "auto").strip() or "auto"
        dtype = self._dtype()

        # 未启用量化：直接按模型原生精度加载，不再尝试 BNB/GPTQ。
        # 此前该情形会无条件落入下方的 bitsandbytes int4 分支，
        # 在无 GPU 的机器上对大模型做一次注定失败的量化尝试。
        if mode in {"", "none", "no", "off", "false", "0"}:
            self.logger.info("量化已关闭（EDGE_QUANTIZATION=%r），按 %s 直接加载", self.cfg.quantization, dtype)
            return AutoModelForCausalLM.from_pretrained(
                model_source,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

        if mode in self._QUANT_DIRECT_MODES:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_source,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            except Exception as exc:  # pragma: no cover
                self.logger.warning("直接加载量化 ckpt 失败，进入 GPTQ/BNB 回退: %s", exc)

        if mode in self._GPTQ_MODES:
            try:
                from transformers import GPTQConfig

                gptq_cfg = GPTQConfig(
                    bits=self.cfg.quant_bits,
                    group_size=self.cfg.quant_group_size,
                )
                return AutoModelForCausalLM.from_pretrained(
                    model_source,
                    quantization_config=gptq_cfg,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            except Exception as exc:  # pragma: no cover
                self.logger.warning("GPTQ/GP32 加载失败，尝试 bitsandbytes int4 兜底: %s", exc)

        try:
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            return AutoModelForCausalLM.from_pretrained(
                model_source,
                quantization_config=bnb_cfg,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception as exc:  # pragma: no cover
            self.logger.warning("bitsandbytes int4 加载失败，回退到模型原生精度: %s", exc)

        return AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> EdgeInferenceResult:
        if not self.ready:
            raise RuntimeError("Edge model not ready")
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Edge tokenizer/model missing")

        generation_tokens = max_new_tokens or self.cfg.max_new_tokens

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)

        inputs = self.tokenizer(
            prompt,
            max_length=self.cfg.max_input_tokens,
            truncation=True,
            return_tensors="pt",
        )

        input_len = inputs["input_ids"].shape[-1]
        generate_kwargs = {
            "input_ids": inputs["input_ids"].to(self.model.device),
            "attention_mask": inputs["attention_mask"].to(self.model.device),
            "max_new_tokens": generation_tokens,
            "do_sample": self.cfg.use_sampling,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if self.cfg.top_k > 0:
            generate_kwargs["top_k"] = self.cfg.top_k

        outputs = self.model.generate(**generate_kwargs)
        generated_ids = outputs.sequences[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        confidence = self._estimate_confidence(outputs.scores)
        return EdgeInferenceResult(
            text=text,
            confidence=confidence,
            used_model=self.model_id,
        )

    @staticmethod
    def _estimate_confidence(scores) -> float:
        if not scores:
            return 0.0

        import torch.nn.functional as F

        window = min(6, len(scores))
        probs = []
        for step_scores in scores[:window]:
            probs.append(float(F.softmax(step_scores[0], dim=-1).max().item()))
        return sum(probs) / len(probs)
