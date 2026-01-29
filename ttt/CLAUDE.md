This project trains and evaluates agents that play tic tac toe.

It uses python and pytorch.

All source code is under src/

The tic tac toe game code is in src/ttt/env.py. When using an env, follow
the Env interface. Most code uses the TttEnv implementation. Ignore
GymEnv.

The agents are in src/ttt/agents/. Agents implement the interface in
src/ttt/agents/agent.py

When making code changes, do the minimal amount of work needed to
fulfil the most recent prompt, then give control back to me. Don't
run tests, linting etc. I'll do all of that.
