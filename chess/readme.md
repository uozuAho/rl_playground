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
- WIP: try POC of https://github.com/dotnet/TorchSharp
    - WIP: try learning andoma board eval
        - use cpu
        - use gpu
- DONE find a fast chess implementation
    - maybe
        - THIS ONE https://github.com/rudzen/ChessLib
            - perf focused, easy-ish api, extensive tests
        - https://github.com/GameDevChef/Chess
            - not a dotnet sln, looks like some game engine
        - https://github.com/PeterHughes/SharpChess
            - quick review: lots of commented out tests, not enough tests
        - https://github.com/ErkrodC/UnityChessLib
            - suspicious: not sure this properly implements bug-free chess
                - code quality bit yuck
                - tests use mocks heavily
                - I don't see all game rules like 2/3 fold repetition
        - [Chess.NET](https://github.com/typedbyte/Chess.NET)
            - 'clean' architecture - AKA inscrutable. Dunno if it's possible to
            play a game with two random agents. Doesn't seem to include legal
            move generation. Kinda interesting I guess. Seems to be written in
            an FP style - immutable, monads.
- NOPE check if ML.NET is feasible
    - try torchsharp first
    - bigger learning curve than torchsharp, worse docs, smaller community
    - maybe too high level?
