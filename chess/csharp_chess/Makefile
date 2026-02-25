build:
	dotnet build

rebuild:
	dotnet build --no-incremental

precommit:
	dotnet csharpier format .
	dotnet test

rank:
	dotnet run --project cschess.tournament
