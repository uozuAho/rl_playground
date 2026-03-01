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

prof-exp-dbg:
	cd cschess.experiments
	dotnet build
	cd ..
	dotnet trace collect --format Speedscope -- dotnet ./cschess.experiments/bin/Debug/net8.0/cschess.experiments.dll

prof-exp-rls:
	cd cschess.experiments
	dotnet build -c Release
	cd ..
	dotnet trace collect --format Speedscope -- dotnet ./cschess.experiments/bin/Release/net8.0/cschess.experiments.dll

prof-clean:
	rm *.speedscope*
	rm *.nettrace*
