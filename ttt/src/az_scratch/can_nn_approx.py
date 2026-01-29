"""Basic check: can the AZ network successfully learn to approximate values?"""

import torch

import ensure_project_path  # noqa python sucks

from torch.optim import Adam

from agents.alphazero import ResNet, _update_net, _net_fwd

import ttt.env as t3

device = "cpu"

model = ResNet(1, 1, device)
optimiser = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# type GameStep = tuple[t3.Board, t3.Player, MctsProbs, float]
steps = [
    [t3.TttEnv.from_str("xx.|oo.|...").board, t3.X, [0, 0, 1, 0, 0, 0, 0, 0, 0], 1.0],
    [t3.TttEnv.from_str("xxx|oo.|...").board, t3.O, [0, 0, 0, 0, 0, 1, 0, 0, 0], -1.0],
]


def pol_val_for(env):
    print("state: ", env.str1d())
    with torch.no_grad():
        pol, val = _net_fwd(model, env, device)
        print("policy: ", pol)
        print("value: ", val.item())


pol_val_for(t3.TttEnv.from_str("xx.|oo.|..."))
pol_val_for(t3.TttEnv.from_str("xxx|oo.|..."))

for i in range(1000):
    pl, vl = _update_net(model, optimiser, steps, device=device)
    if i == 0:
        print("initial losses:", pl, vl)
    elif i == 999:
        print("final losses:", pl, vl)

print("=====================")
print()
print("Did it work?")
print("Expect low losses")

print("This position should have a high value, and the policy should be for action 2:")
pol_val_for(t3.TttEnv.from_str("xx.|oo.|..."))

print()
print("This position should have a low value, and the policy should be for action 5:")
pol_val_for(t3.TttEnv.from_str("xxx|oo.|..."))
