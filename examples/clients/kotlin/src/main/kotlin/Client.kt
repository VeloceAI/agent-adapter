// Call the tool server from Kotlin: MCP with the official SDK, then plain REST.
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.sse.SSE
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.modelcontextprotocol.kotlin.sdk.client.Client
import io.modelcontextprotocol.kotlin.sdk.client.StreamableHttpClientTransport
import io.modelcontextprotocol.kotlin.sdk.types.Implementation
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

fun main() = runBlocking {
    val url = System.getenv("AA_URL") ?: "http://localhost:8000"
    val auth = "Bearer " + (System.getenv("AA_TOKEN") ?: "dev-token")
    val http = HttpClient(CIO) { install(SSE) }

    val client = Client(Implementation(name = "agent-adapter-client", version = "1.0.0"))
    client.connect(StreamableHttpClientTransport(http, "$url/mcp") { header("Authorization", auth) })
    check(client.listTools().tools.any { it.name == "get_weather" }) { "get_weather missing" }
    val result = client.callTool("get_weather", mapOf("city" to "Paris"))
    println("MCP: " + (result.content.first() as TextContent).text)
    client.close()

    val res = http.post("$url/tools/get_weather/invoke") {
        header("Authorization", auth)
        contentType(ContentType.Application.Json)
        setBody("""{"arguments": {"city": "Paris"}}""")
    }
    check(res.status.isSuccess()) { res.bodyAsText() }
    println("REST: " + Json.parseToJsonElement(res.bodyAsText()).jsonObject["result"]!!.jsonPrimitive.content)
    http.close()
}
