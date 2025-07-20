# Chess

...sigh. I'm writing this after half implementing chess bots/libraries/engines
in python (torch) and C++ (libtorch).

pychess is so slow that I went looking for a faster chess implementation that
I could still use with torch. That led me to C++. lc0 is a popular and fast
chess implementation, and pytorch is available through libtorch. I thought I
could vibe code my way through this, but I'm getting memory problems during
training, and I just don't want to learn any more C++. It's awful.

# Options
I've already kinda decided on C#, with the hope that ML.NET is powerful enough
to use for a chess bot. Why C#? Because my ultimate goal is to train a pandemic
agent, and my pandemic implementation is in C#.

# Todo
- check if ML.NET is feasible
- find a fast chess implementation
    - maybe
        - https://github.com/ErkrodC/UnityChessLib
    - no
        - [Chess.NET](https://github.com/typedbyte/Chess.NET)
            - 'clean' architecture - AKA inscrutable. Dunno if it's possible to
            play a game with two random agents. Doesn't seem to include legal
            move generation. Kinda interesting I guess. Seems to be written in
            an FP style - immutable, monads.
