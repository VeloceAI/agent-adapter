# Call the tool server from Ruby: MCP with the official SDK, then plain REST.
require "json"
require "net/http"
require "mcp"

url = ENV.fetch("AA_URL", "http://localhost:8000")
headers = { "Authorization" => "Bearer #{ENV.fetch('AA_TOKEN', 'dev-token')}" }

client = MCP::Client.new(transport: MCP::Client::HTTP.new(url: "#{url}/mcp", headers: headers))
client.connect
names = client.tools.map(&:name)
abort "get_weather missing: #{names}" unless names.include?("get_weather")
result = client.call_tool(name: "get_weather", arguments: { city: "Paris" })
puts "MCP: #{result.dig('result', 'content').map { |part| part['text'] }.join}"

res = Net::HTTP.post(URI("#{url}/tools/get_weather/invoke"),
                     { arguments: { city: "Paris" } }.to_json,
                     headers.merge("Content-Type" => "application/json"))
abort "REST #{res.code}: #{res.body}" unless res.is_a?(Net::HTTPSuccess)
puts "REST: #{JSON.parse(res.body)['result']}"
