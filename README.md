```bash
[load] safetensors: 1.1 ms (290 tensors)
[tokenizer] vocab: 151646, merges: 151387
[load] tokenizer: 203.2 ms (vocab=151646)
[model] H=896 NH=14 NKV=2 NL=24 HD=64 KVD=128 INTER=4864 VOCAB=151936
[load] model FP32: 2792.0 ms

========================================
  Qwen2.5-0.5B Agent (C++ native)
  Commands: /reset  /exit  /temp N
========================================

you > hello! how are you?
ai  > [prefill] 38 tokens... done in 5367 ms
Hello! I'm doing well, thank you for asking. How can I assist you today?
[gen] 19 tokens in 2746 ms (6.9 tok/s)
```
