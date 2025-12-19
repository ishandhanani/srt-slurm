# Accuracy Benchmark

In srt-slurm, users can run different accuracy benchmarks by setting the benchmark section in the config yaml file. Supported benchmarks include `mmlu`, `gpqa` and `longbenchv2`. 

**Note that the `context-length` argument in the config yaml needs to be larger than the `max_tokens` argument of accuracy benchmark.**


## MMLU

For MMLU dataset, the benchmark section in yaml file can be modified in the following way:
```bash
benchmark:
  type: "mmlu"
  num_examples: 200 # Number of examples to run
  max_tokens: 2048 # Max number of output tokens
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
Repeat: 8, mean: 0.812
Scores: ['0.790', '0.820', '0.800', '0.820', '0.820', '0.790', '0.820', '0.840']
====================
Writing report to /tmp/mmlu_deepseek-ai_DeepSeek-R1.html
{'other': np.float64(0.9), 'other:std': np.float64(0.30000000000000004), 'score:std': np.float64(0.36660605559646725), 'stem': np.float64(0.8095238095238095), 'stem:std': np.float64(0.392676726249301), 'humanities': np.float64(0.7428571428571429), 'humanities:std': np.float64(0.4370588154508102), 'social_sciences': np.float64(0.9583333333333334), 'social_sciences:std': np.float64(0.19982631347136331), 'score': np.float64(0.84)}
Writing results to /tmp/mmlu_deepseek-ai_DeepSeek-R1.json
Total latency: 465.618 s
Score: 0.840
Results saved to: /logs/accuracy/mmlu_deepseek-ai_DeepSeek-R1.json
MMLU evaluation complete
```


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


