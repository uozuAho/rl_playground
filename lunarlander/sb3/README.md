# Lunar lander

My implementation of [hugging face RL lunar lander lab](https://huggingface.co/learn/deep-rl-course/en/unit1/hands-on)
[Colab notebook link](https://colab.research.google.com/github/huggingface/deep-rl-class/blob/master/notebooks/unit1/unit1.ipynb#scrollTo=9XaULfDZDvrC)

[Lunar lander docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

Uses
- [stable baselines](https://stable-baselines3.readthedocs.io/en/master/)
    - [PPO algorithm](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#example%5D)
- aigym[box2d]

# Quick start
Install uv

```sh
sudo apt install swig cmake python3-opengl
# maybe install these if you're having uv install issues:
# sudo apt install ffmpeg xvfb python3-dev libfreetype-dev
uv sync
uv run ppo.py               # ppo algo
uv run optuna-a2c.py        # use optuna to tune hyperparams
```
