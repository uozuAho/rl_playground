# Reinforcement learning (RL) playground

This is me trying to learn practical RL.
Inspired by this course: https://huggingface.co/learn/deep-rl-course

At the moment, mostly using
[stable baselines](https://stable-baselines3.readthedocs.io)
PPO algorithm to train agents in toy environments in
[gymnasium](https://gymnasium.farama.org/).

I've copied code from various places, attributed in-place. I think everything
I've grabbed is MIT, so assume that license I guess.


# Other stuff I've done
Every now and then I get interested in AI/RL/machine learning. My trail of
destruction:

## kinda relevant to RL
- [blackjack: tabular(?) learning methods](https://github.com/uozuAho/rl_montecarlo_blackjack)
- [open spiel playground](https://github.com/uozuAho/open_spiel_playground)
    - playing around with open spiel
- [my tensorflow cheatsheets](https://github.com/uozuAho/tensorflow_cheatsheets)
    - tf is hard to (re)learn and use. pytorch seems much nicer

## not very relevant
- [ts_ai2](https://github.com/uozuAho/ts_ai2)
    - a bunch of classical search algorithms in typescript
- [cs ai](https://github.com/uozuAho/cs-ai)
    - same as above, in C#
- [react ai](https://github.com/uozuAho/react_ai)
    - not really sure what i was trying to do here. Use react???



# todo
- reorganise/rename dirs to make exmaples easier to find
    - tabular vs approx
    - proper alg names
    - idea: env/lib/alg.py
        - lib: good, separate by deps
- after:
    - check all run as expected
    - update docs

spaceinvaders/
    rlzoo/dqn.py
blackjack/
    sb3/ppo.py
cartpole
    torch/reinforce.py
chess
    kaggledude/moduleX
    torch
    others
frozenlake
    sb3/ppo.py
    tabular/qlearn.py
lunarlander
    sb3/
        ppo
        optuna-a2c
    torch
ttt
    sb3
        dqn-optuna
        dqn
        mask-ppo
    tabular
        param-search
        tabular.py
