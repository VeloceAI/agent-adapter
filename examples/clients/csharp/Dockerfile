FROM mcr.microsoft.com/dotnet/sdk:10.0
WORKDIR /app
COPY Client.csproj .
RUN dotnet restore
COPY Program.cs .
CMD ["dotnet", "run", "--no-restore"]
