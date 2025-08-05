using csharp;

if (args.Length == 0)
{
    Console.WriteLine("Usage: dotnet run <demo>, eg. dotnet run simple");
    Environment.Exit(1);
}

var toRun = args[0];

switch (toRun)
{
    case "simple":
        Simple.Run();
        break;
    default:
        Console.WriteLine("Unknown Command");
        break;
}
