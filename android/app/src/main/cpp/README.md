# Native edge runtime

This directory is the C++ boundary for the Android edge runtime.

The current implementation is a placeholder until the TinyLlm C++ API is
selected and vendored. Planned components are TinyLlm Gemma E2B INT4,
EmbeddingGemma-300M, FAISS C++, and a small JNI adapter.

Model and index files must be loaded from Context.noBackupFilesDir, never from
a public download directory.
