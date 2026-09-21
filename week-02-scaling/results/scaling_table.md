# Week 2 scaling-sweep results

4 models, only d_model varied (num_layers=2, num_heads=4, block_size=128 fixed - see scaling_configs.py). FLOPs-at-best uses the step of each run's early-stopping-best checkpoint, not total steps trained (see flops.py).

| run | target N | achieved N | final val loss | FLOPs at best | best step | wall-clock (s) |
|---|---|---|---|---|---|---|
| scale_100k | 100,000 | 100,680 | 1.1555 | 1.060e+14 | 42,822 | 1029.9 |
| scale_250k | 250,000 | 244,032 | 1.1062 | 2.634e+14 | 43,920 | 1135.0 |
| scale_500k | 500,000 | 503,720 | 1.0363 | 5.437e+14 | 43,920 | 1698.7 |
| scale_1m | 1,000,000 | 1,007,600 | 0.9697 | 1.088e+15 | 43,920 | 2482.1 |

Total wall-clock across all 4 runs: 6361.0s (106.0 min)
