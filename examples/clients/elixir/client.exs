# Call the tool server from Elixir: MCP with Anubis MCP, then plain REST.
Mix.install([{:anubis_mcp, "~> 2.2"}])

url = System.get_env("AA_URL", "http://localhost:8000")
auth = "Bearer " <> System.get_env("AA_TOKEN", "dev-token")

# Anubis 2.2 does not start its HTTP pool itself when used from a script
{:ok, _} = Finch.start_link(name: Anubis.Finch)

{:ok, _} =
  Anubis.Client.start_link(
    name: :weather,
    transport:
      {:streamable_http, base_url: url, mcp_path: "/mcp", headers: %{"authorization" => auth}},
    client_info: %{"name" => "agent-adapter-client", "version" => "1.0.0"},
    protocol_version: "2025-11-25"
  )

:ok = Anubis.Client.await_ready(:weather, timeout: 10_000)
{:ok, listed} = Anubis.Client.list_tools(:weather)

unless Enum.any?(listed.result["tools"], &(&1["name"] == "get_weather")),
  do: raise("get_weather missing")

{:ok, called} = Anubis.Client.call_tool(:weather, "get_weather", %{"city" => "Paris"})
IO.puts("MCP: " <> hd(called.result["content"])["text"])

:inets.start()
body = JSON.encode!(%{arguments: %{city: "Paris"}})
headers = [{~c"authorization", String.to_charlist(auth)}]
request = {String.to_charlist(url <> "/tools/get_weather/invoke"), headers, ~c"application/json", body}
{:ok, {{_, 200, _}, _, reply}} = :httpc.request(:post, request, [], [])
IO.puts("REST: " <> JSON.decode!(to_string(reply))["result"])
