# Chess

Several chess & chess engine implementations in various languages. Most current
is C#.

# Backstory
...sigh. I'm writing this after half implementing chess bots/libraries/engines
in python (torch) and C++ (libtorch).

pychess is so slow that I went looking for a faster chess implementation that
I could still use with torch. That led me to C++. lc0 is a popular and fast
chess implementation, and pytorch is available through libtorch. I thought I
could vibe code my way through this, but I'm getting memory problems during
training, and I just don't want to learn any more C++. It's awful.
