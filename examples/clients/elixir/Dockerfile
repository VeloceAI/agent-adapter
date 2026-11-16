FROM elixir:1.18-slim
RUN mix local.hex --force && mix local.rebar --force
WORKDIR /app
# fetch and compile the dependency once, at build time (Mix.install caches by its list)
RUN elixir -e 'Mix.install([{:anubis_mcp, "~> 2.2"}])'
COPY client.exs .
CMD ["elixir", "client.exs"]
