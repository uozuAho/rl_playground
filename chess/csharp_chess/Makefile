build:
	dotnet build

rebuild:
	dotnet build --no-incremental

precommit:
	dotnet csharpier format .
	dotnet build
	dotnet test

rank:
	dotnet run --project cschess.tournament

experiment-dbg:
	dotnet run --project cschess.experiments

experiment-rls:
	dotnet run --project cschess.experiments -c Release

exp-trace-dbg:
	dotnet trace collect --format Speedscope -- dotnet ./cschess.experiments/bin/Debug/net8.0/cschess.experiments.dll
