```
Best trial for dqn: #22
  Score: 2704.814
  avg_return: 2852.6
  std_return: 1477.9
  avg_length: 242.6
  avg_max_tile: 230.4
  avg_illegal_attempts: 63.3
  best_params:
    buffer_size: 200000
    batch_size: 128
    gamma: 0.9556791110090774
    lr: 0.00014394430404112647
    target_sync_every: 250
    learn_start: 1000
    learn_every: 8
    eps_end: 0.1488425606123552
    eps_decay_steps: 20000
    grad_clip: 4.3076689475049506
```

```
Best trial for double_dqn: #22
  Score: 2464.240
  avg_return: 2594.8
  std_return: 1305.6
  avg_length: 223.4
  avg_max_tile: 217.6
  avg_illegal_attempts: 55.2
  best_params:
    buffer_size: 20000
    batch_size: 256
    gamma: 0.9798426351600259
    lr: 0.0009768958892956907
    target_sync_every: 100
    learn_start: 1000
    learn_every: 2
    eps_end: 0.17004595587593185
    eps_decay_steps: 50000
    grad_clip: 6.668923925628828
```

```
Best trial for dueling_double_dqn: #6
  Score: 2799.005
  avg_return: 2958.4
  std_return: 1594.0
  avg_length: 245.9
  avg_max_tile: 246.4
  avg_illegal_attempts: 76.6
  best_params:
    buffer_size: 200000
    batch_size: 64
    gamma: 0.9654707747743687
    lr: 0.0029222928006765354
    target_sync_every: 250
    learn_start: 2000
    learn_every: 8
    eps_end: 0.19214154949975376
    eps_decay_steps: 10000
    grad_clip: 1.1189121985227757
```