using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;

namespace csharp;

internal sealed class Mlp : torch.nn.Module
{
    private readonly Sequential _seq;
    public Mlp() : base("MLP")
    {
        _seq = torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        );
        RegisterComponents();
    }

    public torch.Tensor Forward(torch.Tensor x)
    {
        return _seq.forward(x);
    }
}

internal static class Simple
{
    public static void Run()
    {
        TrainTimeReport(torch.CPU);
        Console.WriteLine();
        TrainTimeReport(torch.CUDA);
    }

    private static void TrainTimeReport(torch.Device device)
    {
        var model = new Mlp().to(device);
        var optimizer = torch.optim.Adam(model.parameters());
        var loss_fn = torch.nn.CrossEntropyLoss();

        var x = torch.randn(new long[] { 10000, 100 }, device: device);
        var y = torch.randint(0, 10, new long[] { 10000 }, device: device, dtype: torch.ScalarType.Int64);
        var batch_size = 64;
        var batches_per_epoch = (double)x.shape[0] / batch_size;
        var num_epochs = 5;

        Console.WriteLine($"device={device.type}");
        Console.WriteLine($"X shape: {x.shape}");
        Console.WriteLine($"len(x): {x.shape[0]}");
        Console.WriteLine($"batch size: {batch_size}");

        void Train()
        {
            model.train();
            for (var epoch = 0; epoch < num_epochs; epoch++)
            {
                for (var i = 0; i < x.shape[0]; i += batch_size)
                {
                    var x_batch = x.index_select(0, torch.arange(i, Math.Min(i + batch_size, (int)x.shape[0]), device: device));
                    var y_batch = y.index_select(0, torch.arange(i, Math.Min(i + batch_size, (int)y.shape[0]), device: device));
                    optimizer.zero_grad();
                    var output = model.Forward(x_batch);
                    var loss = loss_fn.forward(output, y_batch);
                    loss.backward();
                    optimizer.step();
                }
            }
        }

        var sw = Stopwatch.StartNew();
        Train();
        sw.Stop();

        var total_time = sw.Elapsed.TotalSeconds;
        var epoch_avg = num_epochs / total_time;
        var batch_avg = (num_epochs * batches_per_epoch) / total_time;
        var sample_avg = (num_epochs * x.shape[0]) / total_time;

        Console.WriteLine($"Epoch time: {total_time:F4} seconds");
        Console.WriteLine($"Epochs per second: {epoch_avg:F2}");
        Console.WriteLine($"Batches per second: {batch_avg:F2}");
        Console.WriteLine($"Samples per second: {sample_avg:F2}");
    }
}
