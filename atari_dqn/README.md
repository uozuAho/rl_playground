# huggingface deep RL course atari DQN lab

https://huggingface.co/learn/deep-rl-course/unit3/hands-on

A simple demo of how to use [RL baselines zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
to train and run an agent with minimal fuss/config.

# quick start
Install uv

```sh
uv install

# train an agent. This saves data to logs/
# tinker with the config file dqn-config.yml
uv run python -m rl_zoo3.train --algo dqn  --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yml

# evaluate the agent
uv run python -m rl_zoo3.enjoy  --algo dqn  --env SpaceInvadersNoFrameskip-v4  --no-render  --n-timesteps 5000  --folder logs/

# run the agent with graphics
uv run python -m rl_zoo3.enjoy  --algo dqn  --env SpaceInvadersNoFrameskip-v4 --n-timesteps 5000  --folder logs/
```
