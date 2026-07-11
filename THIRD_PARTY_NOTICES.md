# Third-party notices

## T2I-CoReBench

The framework-native `t2i_corebench` evaluator is adapted from the benchmark
data contract, evaluation prompt, Qwen/vLLM settings, retry behavior, and score
definitions published by:

- Repository: <https://github.com/KlingAIResearch/T2I-CoReBench>
- Paper: *Easier Painting Than Thinking: Can Text-to-Image Models Set the
  Stage, but Not Direct the Play?* (ICLR 2026)
- Local reference revision used during implementation:
  `ebf7b7a0ac0da088f4664c50012b6b483ac6f02b`

The upstream repository is distributed under the Apache License 2.0. The code
here has been reorganized and modified to use T2IEval's internal model,
evaluator, artifact, cache, and configuration interfaces.
