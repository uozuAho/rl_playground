# learnings
- need epsilon. eps=0 leads to low exploration

# log
10k ep training, 20 steps:

Num winning boards:
452
Average num symmetrics for 10 winning boards:
4.2
Average num symmetrics for top 10 nearly winning boards:
6.4
['1.00', '1.00', '1.00', '0.88', '0.50', '0.00']
['1.00', '0.99', '0.94', '0.88', '0.00', '0.00', '0.00']
['1.00', '0.99', '0.97', '0.75', '0.00']
['1.00', '1.00', '1.00', '0.93', '-0.11', '-0.24']
['1.00', '1.00', '1.00', '1.00', '0.88', '0.23', '0.10']
['1.00', '0.88', '0.75', '0.75', '0.75', '0.50']
['1.00', '0.94', '0.93', '0.75', '0.26', '0.00', '0.00']
['1.00', '1.00', '1.00', '0.97', '0.75', '0.00']
['1.00', '1.00', '0.88', '0.01', '0.00', '0.00', '0.00']
['1.00', '1.00', '0.97', '0.91', '0.50', '0.00', '0.00']


training tabmcts_20k_20...
done in 54.5s
random           (x) vs perfect          (o). 100 games in 0.0s. x wins:   0 (  0.0%), o wins:  86 ( 86.0%), draws:  14 ( 14.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
random           (x) vs tabmcts_20k_20   (o). 100 games in 0.1s. x wins:  17 ( 17.0%), o wins:  75 ( 75.0%), draws:   8 (  8.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs random           (o). 100 games in 0.0s. x wins:  93 ( 93.0%), o wins:   0 (  0.0%), draws:   7 (  7.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs tabmcts_20k_20   (o). 100 games in 0.1s. x wins:  97 ( 97.0%), o wins:   0 (  0.0%), draws:   3 (  3.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_20k_20   (x) vs random           (o). 100 games in 0.2s. x wins:  97 ( 97.0%), o wins:   1 (  1.0%), draws:   2 (  2.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_20k_20   (x) vs perfect          (o). 100 games in 0.2s. x wins:   0 (  0.0%), o wins:  37 ( 37.0%), draws:  63 ( 63.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)

Num boards:
4890
Num winning boards:
565
Average num symmetrics for 10 winning boards (should be 8):
5.3
Average num symmetrics for top 10 nearly winning boards (should be 8):
6.7
Symmetric values for top 10 nearly wins (should all be close to 1.0)
['1.00', '1.00', '1.00', '1.00', '0.96', '0.74', '0.08']
['1.00', '1.00', '1.00', '1.00', '1.00', '0.99', '0.97']
['1.00', '1.00', '1.00', '0.88', '0.81', '0.51', '0.00']
['1.00', '1.00', '1.00', '1.00', '1.00', '0.99', '0.97']
['1.00', '1.00', '1.00', '1.00', '1.00', '0.98', '0.75']
['1.00', '1.00', '1.00', '0.88', '0.69']
['1.00', '1.00', '1.00', '1.00', '1.00', '0.99', '0.98']
['1.00', '1.00', '1.00', '1.00', '0.81', '0.00']
['1.00', '1.00', '1.00', '0.91', '0.08', '0.00', '-0.24']
['1.00', '1.00', '0.94', '0.50', '0.50', '0.50', '-0.24']


training tabmcts_10k_10...
done in 14.8s
random           (x) vs mctsrr10         (o). 100 games in 0.1s. x wins:  27 ( 27.0%), o wins:  63 ( 63.0%), draws:  10 ( 10.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
random           (x) vs perfect          (o). 100 games in 0.0s. x wins:   0 (  0.0%), o wins:  91 ( 91.0%), draws:   9 (  9.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
random           (x) vs tabmcts_10k_10   (o). 100 games in 0.1s. x wins:  21 ( 21.0%), o wins:  72 ( 72.0%), draws:   7 (  7.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
mctsrr10         (x) vs random           (o). 100 games in 0.1s. x wins:  96 ( 96.0%), o wins:   0 (  0.0%), draws:   4 (  4.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
mctsrr10         (x) vs perfect          (o). 100 games in 0.1s. x wins:   0 (  0.0%), o wins:  57 ( 57.0%), draws:  43 ( 43.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
mctsrr10         (x) vs tabmcts_10k_10   (o). 100 games in 0.2s. x wins:  36 ( 36.0%), o wins:  39 ( 39.0%), draws:  25 ( 25.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs random           (o). 100 games in 0.0s. x wins:  98 ( 98.0%), o wins:   0 (  0.0%), draws:   2 (  2.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs mctsrr10         (o). 100 games in 0.1s. x wins:  87 ( 87.0%), o wins:   0 (  0.0%), draws:  13 ( 13.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs tabmcts_10k_10   (o). 100 games in 0.1s. x wins:  84 ( 84.0%), o wins:   0 (  0.0%), draws:  16 ( 16.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_10k_10   (x) vs random           (o). 100 games in 0.1s. x wins:  93 ( 93.0%), o wins:   3 (  3.0%), draws:   4 (  4.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_10k_10   (x) vs mctsrr10         (o). 100 games in 0.2s. x wins:  76 ( 76.0%), o wins:  16 ( 16.0%), draws:   8 (  8.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_10k_10   (x) vs perfect          (o). 100 games in 0.1s. x wins:   0 (  0.0%), o wins:  62 ( 62.0%), draws:  38 ( 38.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)

Num boards:
4027
Num winning boards:
446
Average num symmetrics for 10 winning boards (should be 8):
5.2
Average num symmetrics for top 10 nearly winning boards (should be 8):
5.0
Symmetric values for top 10 nearly wins (should all be close to 1.0)
['1.00', '1.00', '0.88', '0.88', '0.88', '0.50']
['1.00', '1.00', '1.00', '0.99', '0.97', '0.75', '-0.22']
['1.00', '1.00', '1.00', '0.75', '0.37', '0.00']
['1.00', '1.00', '1.00', '0.94', '0.88', '0.50']
['1.00', '0.88', '-0.11']
['1.00', '1.00', '0.81', '0.50']
['1.00', '0.98', '0.88', '0.03']
['1.00', '1.00', '0.75', '0.75', '0.50']
['1.00', '0.98', '0.22']
['1.00', '1.00', '1.00', '0.75', '0.37', '0.00']


training tabmcts_10k_10e0...  e0 = epsilon = 0
done in 19.8s
random           (x) vs mctsrr10         (o). 100 games in 0.1s. x wins:  18 ( 18.0%), o wins:  73 ( 73.0%), draws:   9 (  9.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
random           (x) vs perfect          (o). 100 games in 0.0s. x wins:   0 (  0.0%), o wins:  87 ( 87.0%), draws:  13 ( 13.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
random           (x) vs tabmcts_10k_10e0 (o). 100 games in 0.1s. x wins:  43 ( 43.0%), o wins:  48 ( 48.0%), draws:   9 (  9.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
mctsrr10         (x) vs random           (o). 100 games in 0.1s. x wins:  91 ( 91.0%), o wins:   5 (  5.0%), draws:   4 (  4.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
mctsrr10         (x) vs perfect          (o). 100 games in 0.2s. x wins:   0 (  0.0%), o wins:  59 ( 59.0%), draws:  41 ( 41.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
mctsrr10         (x) vs tabmcts_10k_10e0 (o). 100 games in 0.3s. x wins:  71 ( 71.0%), o wins:  20 ( 20.0%), draws:   9 (  9.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs random           (o). 100 games in 0.0s. x wins:  98 ( 98.0%), o wins:   0 (  0.0%), draws:   2 (  2.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs mctsrr10         (o). 100 games in 0.1s. x wins:  83 ( 83.0%), o wins:   0 (  0.0%), draws:  17 ( 17.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs tabmcts_10k_10e0 (o). 100 games in 0.1s. x wins: 100 (100.0%), o wins:   0 (  0.0%), draws:   0 (  0.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_10k_10e0 (x) vs random           (o). 100 games in 0.1s. x wins:  78 ( 78.0%), o wins:  17 ( 17.0%), draws:   5 (  5.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_10k_10e0 (x) vs mctsrr10         (o). 100 games in 0.2s. x wins:  33 ( 33.0%), o wins:  61 ( 61.0%), draws:   6 (  6.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_10k_10e0 (x) vs perfect          (o). 100 games in 0.1s. x wins:   0 (  0.0%), o wins:  86 ( 86.0%), draws:  14 ( 14.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)

Num boards:
277
Num winning boards:
50
Average num symmetrics for 10 winning boards (should be 8):
1.9
Average num symmetrics for top 10 nearly winning boards (should be 8):
1.2
Symmetric values for top 10 nearly wins (should all be close to 1.0)
['1.00']
['1.00']
['1.00']
['1.00']
['1.00']
['0.95', '0.24']
['0.95', '0.00']
['0.95']
['0.95']
['0.95']


training tabmcts_10k_10lr.1... lr = 0.1
done in 14.5s
random           (x) vs mctsrr10         (o). 100 games in 0.1s. x wins:  17 ( 17.0%), o wins:  73 ( 73.0%), draws:  10 ( 10.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
random           (x) vs perfect          (o). 100 games in 0.0s. x wins:   0 (  0.0%), o wins:  92 ( 92.0%), draws:   8 (  8.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
random           (x) vs tabmcts_10k_10lr.1 (o). 100 games in 0.1s. x wins:  19 ( 19.0%), o wins:  68 ( 68.0%), draws:  13 ( 13.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
mctsrr10         (x) vs random           (o). 100 games in 0.1s. x wins:  93 ( 93.0%), o wins:   4 (  4.0%), draws:   3 (  3.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
mctsrr10         (x) vs perfect          (o). 100 games in 0.1s. x wins:   0 (  0.0%), o wins:  62 ( 62.0%), draws:  38 ( 38.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
mctsrr10         (x) vs tabmcts_10k_10lr.1 (o). 100 games in 0.2s. x wins:  39 ( 39.0%), o wins:  38 ( 38.0%), draws:  23 ( 23.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs random           (o). 100 games in 0.0s. x wins:  93 ( 93.0%), o wins:   0 (  0.0%), draws:   7 (  7.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs mctsrr10         (o). 100 games in 0.1s. x wins:  80 ( 80.0%), o wins:   0 (  0.0%), draws:  20 ( 20.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
perfect          (x) vs tabmcts_10k_10lr.1 (o). 100 games in 0.1s. x wins:  84 ( 84.0%), o wins:   0 (  0.0%), draws:  16 ( 16.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_10k_10lr.1 (x) vs random           (o). 100 games in 0.1s. x wins:  97 ( 97.0%), o wins:   0 (  0.0%), draws:   3 (  3.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_10k_10lr.1 (x) vs mctsrr10         (o). 100 games in 0.2s. x wins:  75 ( 75.0%), o wins:  16 ( 16.0%), draws:   9 (  9.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)
tabmcts_10k_10lr.1 (x) vs perfect          (o). 100 games in 0.1s. x wins:   0 (  0.0%), o wins:  59 ( 59.0%), draws:  41 ( 41.0%), x illegal:   0 (  0.0%), o illegal:   0 (  0.0%)

Num boards:
3625
Num winning boards:
388
Average num symmetrics for 10 winning boards (should be 8):
4.1
Average num symmetrics for top 10 nearly winning boards (should be 8):
5.2
Symmetric values for top 10 nearly wins (should all be close to 1.0)
['1.00', '0.83', '0.32', '0.10', '0.10', '0.00']
['1.00', '0.60', '0.10', '0.00', '-0.06']
['1.00', '0.97', '0.10', '0.10']
['1.00', '0.57', '0.34', '0.27']
['1.00', '1.00', '0.98', '0.10', '0.01', '0.01']
['1.00', '0.56', '0.50', '0.10', '0.00']
['1.00', '1.00', '0.98', '0.10', '0.01', '0.01']
['1.00', '0.87', '0.87', '0.68', '0.38', '0.00', '0.00']
['1.00', '0.01', '0.01', '0.00']
['1.00', '0.96', '0.78', '0.44', '0.17']
