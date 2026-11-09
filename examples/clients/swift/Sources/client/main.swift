// Call the tool server from Swift: MCP with the official SDK, then plain REST.
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import MCP

struct Failure: Error { let message: String }

let env = ProcessInfo.processInfo.environment
let base = env["AA_URL"] ?? "http://localhost:8000"
let auth = "Bearer " + (env["AA_TOKEN"] ?? "dev-token")

let client = Client(name: "agent-adapter-client", version: "1.0.0")
let transport = HTTPClientTransport(
    endpoint: URL(string: base + "/mcp")!,
    requestModifier: { request in
        var request = request
        request.setValue(auth, forHTTPHeaderField: "Authorization")
        return request
    })
try await client.connect(transport: transport)
let (tools, _) = try await client.listTools()
guard tools.contains(where: { $0.name == "get_weather" }) else { throw Failure(message: "get_weather missing") }
let (content, _) = try await client.callTool(name: "get_weather", arguments: ["city": "Paris"])
guard case .text(let text, _, _) = content.first else { throw Failure(message: "no text content") }
print("MCP: \(text)")
await client.disconnect()

var request = URLRequest(url: URL(string: base + "/tools/get_weather/invoke")!)
request.httpMethod = "POST"
request.setValue(auth, forHTTPHeaderField: "Authorization")
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.httpBody = try JSONSerialization.data(withJSONObject: ["arguments": ["city": "Paris"]])
let (data, response) = try await URLSession.shared.data(for: request)
guard (response as? HTTPURLResponse)?.statusCode == 200,
    let reply = try JSONSerialization.jsonObject(with: data) as? [String: Any],
    let result = reply["result"] as? String
else { throw Failure(message: "REST failed") }
print("REST: \(result)")
