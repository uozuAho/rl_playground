using cschess.experiments;

Console.WriteLine(string.Join(",", args));

var fenScoresPath = args[0];

ValueNetworkTrainer.TrainAndTestValueNetwork(fenScoresPath);
