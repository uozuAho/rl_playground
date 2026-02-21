This is a C# project for training and evaluating
chess agents. It uses TorchSharp as the machine
learning library.

Source code is in project directories - these all
start with "cschess". The one exception is the
chess game implementation which lives under submodules.
It is wrapped by the IChessGame interface in cschess.game.

Chess agents are in cschess.agents.

When doing work, do the minimum required to fulfil
the prompt. Don't run tests or linting etc., I'll
do all of that.
