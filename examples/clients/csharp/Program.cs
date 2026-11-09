// Call the tool server from C#: MCP with the official SDK, then plain REST.
using System.Net.Http.Json;
using System.Text.Json;
using ModelContextProtocol.Client;

var url = Environment.GetEnvironmentVariable("AA_URL") ?? "http://localhost:8000";
var token = Environment.GetEnvironmentVariable("AA_TOKEN") ?? "dev-token";
var auth = $"Bearer {token}";

var transport = new HttpClientTransport(new HttpClientTransportOptions
{
    Endpoint = new Uri($"{url}/mcp"),
    AdditionalHeaders = new Dictionary<string, string> { ["Authorization"] = auth },
});
await using var client = await McpClient.CreateAsync(transport);
var tools = await client.ListToolsAsync();
if (!tools.Any(t => t.Name == "get_weather")) throw new Exception("get_weather missing");
var result = await client.CallToolAsync("get_weather", new Dictionary<string, object?> { ["city"] = "Paris" });
Console.WriteLine($"MCP: {result.Content.OfType<ModelContextProtocol.Protocol.TextContentBlock>().First().Text}");

using var http = new HttpClient();
http.DefaultRequestHeaders.Add("Authorization", auth);
var res = await http.PostAsJsonAsync($"{url}/tools/get_weather/invoke", new { arguments = new { city = "Paris" } });
res.EnsureSuccessStatusCode();
var body = await res.Content.ReadFromJsonAsync<JsonElement>();
Console.WriteLine($"REST: {body.GetProperty("result").GetString()}");
