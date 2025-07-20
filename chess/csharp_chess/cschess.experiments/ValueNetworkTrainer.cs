using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using TorchSharp.Tensor;
using TorchSharp.Modules;
using static TorchSharp.torch;
using System.Threading.Tasks;

namespace cschess.experiments
{
    public class ValueNetworkTrainer
    {
        // Mean Squared Error
        public static double MeanSquaredError(double[] yTrue, double[] yPred)
        {
            return yTrue.Zip(yPred, (a, b) => Math.Pow(a - b, 2)).Average();
        }

        // Mean Absolute Error
        public static double MeanAbsoluteError(double[] yTrue, double[] yPred)
        {
            return yTrue.Zip(yPred, (a, b) => Math.Abs(a - b)).Average();
        }

        // Pearson correlation
        public static (double correlation, double pValue) PearsonR(double[] x, double[] y)
        {
            double meanX = x.Average();
            double meanY = y.Average();
            double numerator = x.Zip(y, (a, b) => (a - meanX) * (b - meanY)).Sum();
            double sumSqX = x.Select(a => Math.Pow(a - meanX, 2)).Sum();
            double sumSqY = y.Select(b => Math.Pow(b - meanY, 2)).Sum();
            double denominator = Math.Sqrt(sumSqX * sumSqY);
            if (denominator == 0) return (0.0, 1.0);
            double correlation = numerator / denominator;
            int n = x.Length;
            double pValue;
            if (n <= 2)
                pValue = 1.0;
            else
            {
                double tStat = correlation * Math.Sqrt((n - 2) / (1 - Math.Pow(correlation, 2) + 1e-8));
                pValue = 2 * (1 - Math.Abs(tStat) / (Math.Abs(tStat) + Math.Sqrt(n - 2)));
                pValue = Math.Max(0.0, Math.Min(1.0, pValue));
            }
            return (correlation, pValue);
        }

        // Normalize Michniew score
        public static double NormalizeMichniewScore(int score)
        {
            return Math.Tanh(score / 3000.0);
        }

        // Generate data
        public static (List<ChessGame>, List<double>) GenerateData(int nPositions)
        {
            var positions = new List<ChessGame>();
            var scores = new List<double>();
            var rand = new Random();
            for (int i = 0; i < nPositions; i++)
            {
                var game = new ChessGame();
                int nMoves = rand.Next(5, 51);
                for (int j = 0; j < nMoves; j++)
                {
                    var legalMoves = game.LegalMoves();
                    if (legalMoves.Count == 0 || game.IsGameOver()) break;
                    var move = legalMoves[rand.Next(legalMoves.Count)];
                    game.Step(move);
                }
                positions.Add(game);
                scores.Add(NormalizeMichniewScore(Michniew.EvaluateBoard(game.Board)));
            }
            return (positions, scores);
        }

        // Read data file
        public static (List<ChessGame>, List<double>) ReadDataFile(string path)
        {
            var positions = new List<ChessGame>();
            var scores = new List<double>();
            foreach (var line in File.ReadLines(path))
            {
                var parts = line.Split(',');
                positions.Add(new ChessGame(parts[0]));
                scores.Add(double.Parse(parts[1]));
            }
            return (positions, scores);
        }

        // Train value network
        public static (List<double> trainLosses, List<double> valLosses) TrainValueNetwork(
            Module valueNetwork,
            List<ChessGame> trainPositions,
            List<double> trainTargets,
            List<ChessGame> testPositions,
            List<double> testTargets,
            int epochs = 100,
            int batchSize = 32,
            double lr = 1e-3,
            Device device = null)
        {
            valueNetwork.to(device);
            valueNetwork.train();
            var optimizer = torch.optim.Adam(valueNetwork.parameters(), lr: lr);
            var criterion = nn.MSELoss();
            var trainPositionsT = torch.tensor(trainPositions.Select(b => b.StateNP()).ToArray(), device: device);
            var trainTargetsT = torch.tensor(trainTargets.ToArray(), dtype: ScalarType.Float32, device: device);
            var testPositionsT = torch.tensor(testPositions.Select(b => b.StateNP()).ToArray(), device: device);
            var testTargetsT = torch.tensor(testTargets.ToArray(), dtype: ScalarType.Float32, device: device);
            var trainLosses = new List<double>();
            var valLosses = new List<double>();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var indices = torch.randperm(trainPositionsT.shape[0], device: device);
                var epochLosses = new List<double>();
                for (int i = 0; i < trainPositionsT.shape[0]; i += batchSize)
                {
                    var batchIndices = indices.slice(0, i, Math.Min(i + batchSize, trainPositionsT.shape[0]));
                    var batchPositions = trainPositionsT.index_select(0, batchIndices);
                    var batchTargets = trainTargetsT.index_select(0, batchIndices);
                    optimizer.zero_grad();
                    var predictions = valueNetwork.forward(batchPositions).squeeze();
                    var loss = criterion.forward(predictions, batchTargets);
                    loss.backward();
                    optimizer.step();
                    epochLosses.Add(loss.ToSingle());
                }
                trainLosses.Add(epochLosses.Average());
                valueNetwork.eval();
                using (torch.no_grad())
                {
                    var valPredictions = valueNetwork.forward(testPositionsT).squeeze();
                    var valLoss = criterion.forward(valPredictions, testTargetsT).ToSingle();
                    valLosses.Add(valLoss);
                }
                valueNetwork.train();
            }
            return (trainLosses, valLosses);
        }

        // Evaluate accuracy
        public static Dictionary<string, double> EvaluateAccuracy(double[] networkScores, double[] michniewScores)
        {
            var mse = MeanSquaredError(michniewScores, networkScores);
            var mae = MeanAbsoluteError(michniewScores, networkScores);
            var (correlation, pValue) = PearsonR(networkScores, michniewScores);
            return new Dictionary<string, double>
            {
                {"mse", mse},
                {"mae", mae},
                {"correlation", correlation},
                {"p_value", pValue},
                {"rmse", Math.Sqrt(mse)}
            };
        }

        // Main training/testing entry point
        public static (Module valueNetwork, Dictionary<string, double> metrics) TrainAndTestValueNetwork(
            string datasetPath = null,
            int? datasetSize = null)
        {
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            List<ChessGame> positions;
            List<double> values;
            if (datasetPath != null)
            {
                (positions, values) = ReadDataFile(datasetPath);
            }
            else if (datasetSize.HasValue)
            {
                (positions, values) = GenerateData(datasetSize.Value);
            }
            else
            {
                throw new Exception("Either provide a dataset file or data size");
            }
            int splitIdx = (int)(0.8 * positions.Count);
            var trainPositions = positions.Take(splitIdx).ToList();
            var trainTargets = values.Take(splitIdx).ToList();
            var testPositions = positions.Skip(splitIdx).ToList();
            var testTargets = values.Skip(splitIdx).ToList();
            var valueNetwork = new ValueNetwork();
            var (trainLosses, valLosses) = TrainValueNetwork(
                valueNetwork,
                trainPositions,
                trainTargets,
                testPositions,
                testTargets,
                epochs: 50,
                batchSize: 64,
                lr: 1e-3,
                device: device
            );
            valueNetwork.eval();
            using (torch.no_grad())
            {
                var testPositionsT = torch.tensor(testPositions.Select(b => b.StateNP()).ToArray(), device: device);
                var networkScoresT = valueNetwork.forward(testPositionsT).squeeze();
                var networkScores = networkScoresT.cpu().data<float>().ToArray();
                var metrics = EvaluateAccuracy(networkScores, testTargets.ToArray());
                return (valueNetwork, metrics);
            }
        }
    }
}
