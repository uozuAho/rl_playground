using cschess.agents;

// train value network
// using cschess.experiments;
// Console.WriteLine(string.Join(",", args));
// var fenScoresPath = args[0];
// ValueNetworkTrainer.TrainAndTestValueNetwork(fenScoresPath);

var nnAgent = new GreedyNnAgent();
var randomAgent = new RandomAgent();
nnAgent.TrainAgainst(randomAgent, 10);

