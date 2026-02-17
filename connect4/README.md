# Connect four

```sh
make init
make test
make train
# see makefile for more
```

# todo
- azmp
    - training efficiency (agent strength vs training time):
        - try with/without discard on weight update
        - try with/without valid action masking
    - can it reach azfs agent strength?
        - azfs pre-trained, net 9 128 mcts 10 100% wins against random and first legal, as X and O

## maybe
- save net/train saved net
- try different net/conv sizes
- maybe: perf: minimised steps discarded when discard = True

# notes
- azmp training notes
    - avg 16 steps/game
    - perf
        - net 4 64 mcts 60:  36k steps trained in 180sec = 200 steps/sec ~= 12 games/sec
        - net 9 128 mcts 60: 18k steps trained in 180sec = 100 steps/sec ~= 6 games/sec (approx 3x azfs)
    - training efficiency
- training performance of https://github.com/foersterrobert/AlphaZero
    - pre-trained model, mcts 10 sims, 100% vs random and first legal, X and O
    - from azfs video transcript: trained for 8 iterations, took a few hours
    - from code: 24000 games, 600 sims for training, temp 1.25, C 2, alpha 0.3, eps 0.25
    - training on my machine
        - config 60 mcts, net 9, 128, cuda, 4 epoch, 128 batch, t 1.25, alpha 0.3, eps 0.25,
          lr 0.001, weight decay 1e-4, eval 10 mcts
        - trains at approx 2 games/sec
        - at 5000 games
            - nearly 100 vs rng, as x and o. note it still loses some after 6000 games
            - still weak vs first legal, esp as o. often loses 100% as o
        - at 8000 games
            - strong vs rng but still loses some
            - x vs first legal strong but variable
            - o vs first legal very weak, loses most
        - trained 8600 games
            as X vs rng, WLD:  17 3 0
            as O vs rng, WLD:  18 2 0
            as X vs first legal, WLD:  20 0 0
            as O vs first legal, WLD:  19 1 0
            8500 was even stronger
        - very strong at this point. first perfect eval at 9100
        - still seeing poor perf against first legal after 9800
        - can still see pretty bad regressions later, eg
            trained 10400 games
                as X vs rng, WLD:  17 3 0
                as O vs rng, WLD:  15 5 0
                as X vs first legal, WLD:  6 14 0
                as O vs first legal, WLD:  18 2 0

