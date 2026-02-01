This project trains and evaluates agents that play connect four.

It uses python, numpy and pytorch.

All source code is under src/
The connect four game implementation is in src/env/connect4.py
Game playing agents are in src/agents

Code style is mostly functional. Prefer functional code over object-oriented.
Classes are mainly used as data bags, with first class functions to operate
on those classes. Type hints are used as much as possible.

When making code changes, do the minimal amount of work needed to
fulfil the most recent prompt, then give control back to me. Don't
run tests, linting etc. I'll do all of that.
