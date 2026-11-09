// Call the tool server from Java: MCP with the official SDK, then plain REST.
import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.transport.HttpClientStreamableHttpTransport;
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest;
import io.modelcontextprotocol.spec.McpSchema.TextContent;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Map;

public class Client {
    public static void main(String[] args) throws Exception {
        String url = System.getenv().getOrDefault("AA_URL", "http://localhost:8000");
        String auth = "Bearer " + System.getenv().getOrDefault("AA_TOKEN", "dev-token");

        var transport = HttpClientStreamableHttpTransport.builder(url).endpoint("/mcp")
                .requestBuilder(HttpRequest.newBuilder().header("Authorization", auth)).build();
        try (var client = McpClient.sync(transport).build()) {
            client.initialize();
            if (client.listTools().tools().stream().noneMatch(t -> t.name().equals("get_weather")))
                throw new IllegalStateException("get_weather missing");
            var result = client.callTool(
                    CallToolRequest.builder("get_weather").arguments(Map.of("city", "Paris")).build());
            System.out.println("MCP: " + ((TextContent) result.content().get(0)).text());
        }

        var request = HttpRequest.newBuilder(URI.create(url + "/tools/get_weather/invoke"))
                .header("Authorization", auth).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{\"arguments\": {\"city\": \"Paris\"}}")).build();
        var response = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) throw new IllegalStateException(response.body());
        var json = tools.jackson.databind.json.JsonMapper.builder().build().readTree(response.body());
        System.out.println("REST: " + json.get("result").asString());
    }
}
