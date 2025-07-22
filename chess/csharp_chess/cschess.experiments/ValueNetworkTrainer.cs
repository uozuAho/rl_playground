using System.Diagnostics;
using cschess.game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace cschess.experiments;

public sealed class ValueNetwork : nn.Module
{
    private nn.Module _conv1, _conv2, _conv3;
    private nn.Module _fc1, _fc2, _fc3;
    private nn.Module _dropout;

    public ValueNetwork() : base("ValueNetwork")
    {
        _conv1 = nn.Conv2d(8, 32, 3, stride: 1, padding: 1);
        _conv2 = nn.Conv2d(32, 64, 3, stride: 1, padding: 1);
        _conv3 = nn.Conv2d(64, 32, 3, stride: 1, padding: 1);
        _fc1 = nn.Linear(32 * 8 * 8, 128);
        _fc2 = nn.Linear(128, 64);
        _fc3 = nn.Linear(64, 1);
        _dropout = nn.Dropout(0.3);

        RegisterComponents();
    }

    public Tensor Forward(Tensor x)
    {
        x = nn.functional.relu(((Conv2d)_conv1).forward(x));
        x = nn.functional.relu(((Conv2d)_conv2).forward(x));
        x = nn.functional.relu(((Conv2d)_conv3).forward(x));
        x = x.view(x.shape[0], -1);
        x = nn.functional.relu(((Linear)_fc1).forward(x));
        x = ((Dropout)_dropout).forward(x);
        x = nn.functional.relu(((Linear)_fc2).forward(x));
        x = nn.functional.tanh(((Linear)_fc3).forward(x));

        return x;
    }
}

public class ValueNetworkTrainer
{
    public static double MeanSquaredError(double[] yTrue, float[] yPred)
    {
        return yTrue.Zip(yPred, (a, b) => Math.Pow(a - b, 2)).Average();
    }

    public static double MeanAbsoluteError(double[] yTrue, float[] yPred)
    {
        return yTrue.Zip(yPred, (a, b) => Math.Abs(a - b)).Average();
    }

    public static (double correlation, double pValue) PearsonR(float[] x, double[] y)
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

    public static double NormalizeMichniewScore(int score)
    {
        return Math.Tanh(score / 3000.0);
    }

    public static (List<CodingAdventureChessGame>, List<double>) ReadDataFile(string path)
    {
        var positions = new List<CodingAdventureChessGame>();
        var scores = new List<double>();
        foreach (var line in File.ReadLines(path))
        {
            var parts = line.Split(',');
            positions.Add(CodingAdventureChessGame.FromFen(parts[0]));
            scores.Add(double.Parse(parts[1]));
        }
        return (positions, scores);
    }

    public static Tensor Board2Tensor(CodingAdventureChessGame game)
    {
        var tensor = zeros(8, 8, 8, float32);
        for (var i = 0; i < 64; ++i) {
            var row = i / 8;
            var col = i % 8;
            var pieceType = game.PieceAt(i);
            if (!pieceType.HasValue) continue;
            var color = game.ColorAt(i);
            Debug.Assert(color != Color.None);
            var pieceIdx = (int)pieceType - 1;
            tensor[pieceIdx][row][col] = (float)color;
        }
        var fullmove = game.FullmoveCount();
        if (fullmove > 0) {
            tensor[6].fill_(1.0f / fullmove);
        }
        if (game.Turn() == Color.White) {
            tensor[6][0].fill_(1.0f);
        } else {
            tensor[6][0].fill_(-1.0f);
        }
        tensor[7].fill_(1.0f);
        return tensor;
    }

    // Train value network
    public static (List<double> trainLosses, List<double> valLosses) TrainValueNetwork(
        ValueNetwork valueNetwork,
        List<CodingAdventureChessGame> trainPositions,
        List<double> trainTargets,
        List<CodingAdventureChessGame> testPositions,
        List<double> testTargets,
        int epochs = 100,
        int batchSize = 32,
        double lr = 1e-3,
        Device device = null)
    {
        valueNetwork.to(device);
        valueNetwork.train();
        // var testtt = Board2Tensor(trainPositions.First());
        var optimizer = optim.Adam(valueNetwork.parameters(), lr: lr);
        var criterion = nn.MSELoss();
        var trainPositionsT = stack(trainPositions.Select(Board2Tensor).ToArray());
        var trainTargetsT = tensor(trainTargets, dtype: ScalarType.Float32, device: device);
        var testPositionsT = stack(testPositions.Select(Board2Tensor).ToArray());
        var testTargetsT = tensor(testTargets, dtype: ScalarType.Float32, device: device);
        var trainLosses = new List<double>();
        var valLosses = new List<double>();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Console.WriteLine($"Epoch {epoch}");

            var indices = randperm(trainPositionsT.shape[0], device: device);
            var epochLosses = new List<double>();
            for (int i = 0; i < trainPositionsT.shape[0]; i += batchSize)
            {
                var batchIndices = indices.slice(0, i, Math.Min(i + batchSize, trainPositionsT.shape[0]), 1);
                var batchPositions = trainPositionsT.index_select(0, batchIndices);
                var batchTargets = trainTargetsT.index_select(0, batchIndices);
                optimizer.zero_grad();
                var predictions = valueNetwork.Forward(batchPositions).squeeze();
                var loss = criterion.forward(predictions, batchTargets);
                loss.backward();
                optimizer.step();
                epochLosses.Add(loss.ToSingle());
            }
            trainLosses.Add(epochLosses.Average());
            valueNetwork.eval();
            using (no_grad())
            {
                var valPredictions = valueNetwork.Forward(testPositionsT).squeeze();
                var valLoss = criterion.forward(valPredictions, testTargetsT).ToSingle();
                valLosses.Add(valLoss);
            }
            valueNetwork.train();
        }
        return (trainLosses, valLosses);
    }

    // Evaluate accuracy
    public static Dictionary<string, double> EvaluateAccuracy(float[] networkScores, double[] michniewScores)
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

    public static void TrainAndTestValueNetwork(string datasetPath)
    {
        var device = cuda.is_available() ? CUDA : CPU;

        Console.WriteLine($"Training on data read from {datasetPath}");

        var (positions, values) = ReadDataFile(datasetPath);

        Console.WriteLine($"Positions: {positions.Count}");

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
        using (no_grad())
        {
            var testPositionsT = stack(testPositions.Select(Board2Tensor));
            var networkScoresT = valueNetwork.Forward(testPositionsT).squeeze();
            var networkScores = networkScoresT.cpu().data<float>().ToArray();
            var metrics = EvaluateAccuracy(networkScores, testTargets.ToArray());

            foreach (var (key, value) in metrics)
            {
                Console.WriteLine($"{key}: {value}");
            }
        }
    }
}
