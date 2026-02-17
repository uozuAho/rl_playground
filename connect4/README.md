# Connect four

Mainly focused on azmp - AlphaZero MultiProcessing - training AZ as fast as possible
using multiple processses.

```sh
make init
make test
make train
# see makefile for more
```

# notes
- azmp training notes
    - avg 16 steps/game
    - masking invalid actions during training does seem to increase efficiency
    - discarding steps after a weight update slows training 3x, and doesn't
      noticeably affect efficiency
    - perf
        - net 4 64 mcts 60:  36k steps trained in 180sec = 200 steps/sec ~= 12 games/sec
        - net 9 128 mcts 60: 18k steps trained in 180sec = 100 steps/sec ~= 6 games/sec (approx 3x azfs)
## agent strength vs training
- azmp net 1 2, mcts 60 train, 10 eval
    - 300k steps in 800sec, 375 steps/sec ~ 23 games/sec
    - at 100k steps
        - as X: 70/30% win/loss vs mcts rr 100
        - as O: 50/50
    - at 300k steps, X: 80/20, O: 60/40
- azmp net 4 64, mcts 60 train, 10 eval
    - 300k steps in 1400sec, 214 steps/sec ~ 13 games/sec
    - at 100k steps
        - as X: 85/10% win/loss vs mcts rr 100
        - as O: 70/30
    - at 300k steps, X: 95/5, O: 90/10
- azfs pre-trained, net 9 128 mcts 10: https://github.com/foersterrobert/AlphaZero
    - opponent        win/loss% as X     win/loss% as O
    - random                  100/?              100/?
    - first legal             100/?              100/?
    - mcts rr 60               90/10              86/14
    - mcts rr 100              60/36              76/16
