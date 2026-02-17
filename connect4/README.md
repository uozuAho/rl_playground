# Connect four

Mainly focused on azmp - AlphaZero MultiProcessing - training AZ as fast as possible
using multiple processses.

```sh
make init
make test
make train
# see makefile for more
```

# todo
- azmp
    - WIP can it reach azfs agent strength?
        - todo
            - WIP assess asymp performance of 4 64 and 1 2 agents
                - save bigger 4 64 pic from downloads
                - run 1 2 to asymp
                - review pics
            - assess early training perf of above
        - note: azfs pre-trained, net 9 128 mcts 10:
            - opponent        win/loss% as X     win/loss% as O
            - random                  100/?              100/?
            - first legal             100/?              100/?
            - mcts rr 60               90/10              86/14
            - mcts rr 100              60/36              76/16


## maybe
- save net/train saved net

# notes
- azmp training notes
    - avg 16 steps/game
    - masking invalid actions during training does seem to increase efficiency
    - discarding steps after a weight update slows training 3x, and doesn't
      noticeably affect efficiency
    - perf
        - net 4 64 mcts 60:  36k steps trained in 180sec = 200 steps/sec ~= 12 games/sec
        - net 9 128 mcts 60: 18k steps trained in 180sec = 100 steps/sec ~= 6 games/sec (approx 3x azfs)
