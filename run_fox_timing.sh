#!/bin/bash
exec /workspace/target/release/fox serve \
    --model-path /workspace/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf \
    --models-dir /workspace/models \
    --max-batch-size 4 \
    --max-context-len 4096 \
    --flash-attn true \
    --chunked-prefill-tokens 512 \
    --port 8080 \
    > /workspace/fox_timing.log 2>&1
