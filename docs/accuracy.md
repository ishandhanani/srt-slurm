# Accuracy Benchmark

In srt-slurm, users can run different accuracy benchmarks by setting the benchmark section in the config yaml file. Supported benchmarks include `mmlu`, `gpqa` and `longbenchv2`. 

**Note that the `context-length` argument in the config yaml needs to be larger than the `max_tokens` argument of accuracy benchmark.**


## MMLU

For MMLU dataset, the benchmark section in yaml file can be modified in the following way:
```bash
benchmark:
  type: "mmlu"
  num_examples: 200 # Number of examples to run
  max_tokens: 8192 # Max number of output tokens.
  repeat: 8 # Number of repetition
  num_threads: 512 # Number of parallel threads for running benchmark
```
 
Then launch the script as usual:
```bash
srtctl apply -f config.yaml
```

After finishing benchmarking, the `benchmark.out` will contain the results of accuracy:
```
====================
Repeat: 8, mean: 0.895
Scores: ['0.905', '0.895', '0.900', '0.880', '0.905', '0.890', '0.890', '0.895']
====================
Writing report to /tmp/mmlu_deepseek-ai_DeepSeek-R1.html
{'other': np.float64(0.9361702127659575), 'other:std': np.float64(0.24444947432076722), 'score:std': np.float64(0.3065534211193866), 'stem': np.float64(0.9285714285714286), 'stem:std': np.float64(0.25753937681885636), 'humanities': np.float64(0.8064516129032258), 'humanities:std': np.float64(0.3950789907714804), 'social_sciences': np.float64(0.9387755102040817), 'social_sciences:std': np.float64(0.23974163519328023), 'score': np.float64(0.895)}
Writing results to /tmp/mmlu_deepseek-ai_DeepSeek-R1.json
Total latency: 754.457 s
Score: 0.895
Results saved to: /logs/accuracy/mmlu_deepseek-ai_DeepSeek-R1.json
MMLU evaluation complete
```

**Note: `max-tokens` should be large enough to reach expected accuracy. For deepseek-r1-fp4 model, `max-tokens=8192` can reach expected accuracy 0.895, while `max-tokens=2048` can only score at 0.81.**


## GPQA
For GPQA dataset, the benchmark section in yaml file can be modified in the following way:
```bash
benchmark:
  type: "gpqa"
  num_examples: 198 # Number of examples to run
  max_tokens: 65536 # We need a larger output token number for GPQA
  repeat: 8 # Number of repetition
  num_threads: 128 # Number of parallel threads for running benchmark
```
The `context-length` argument here should be set to a value larger than `max_tokens`.


## LongBench-V2
To be updated


