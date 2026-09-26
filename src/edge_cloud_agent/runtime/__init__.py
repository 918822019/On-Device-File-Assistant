"""端云推理运行时与编排层。

- routing:           纯路由策略(RoutingPolicy / EdgeFallbackPolicy)
- agent:             EdgeCloudOrchestrator——端侧优先、云端兜底的编排与降级
- edge_runtime:      端侧 LLM(gemma-4-E2B-it)加载与推理
- embedding_runtime: 端侧 embedding(embeddinggemma-300m)向量化,业务线共用
- cloud_client:      云端 OpenAI 兼容接口薄封装

依赖方向:runtime → config / common;业务线可依赖 runtime.embedding_runtime;
routers 与 main 依赖 runtime.agent。本包 __init__ 不做再导出,
各模块按需显式导入(避免包 import 触发 torch/transformers 加载)。
"""
