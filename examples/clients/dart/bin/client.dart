// Call the tool server from Dart (and Flutter): MCP with mcp_dart, then plain REST.
import 'dart:convert';
import 'dart:io';

import 'package:mcp_dart/mcp_dart.dart';

Future<void> main() async {
  final url = Platform.environment['AA_URL'] ?? 'http://localhost:8000';
  final auth = 'Bearer ${Platform.environment['AA_TOKEN'] ?? 'dev-token'}';

  final client = McpClient(Implementation(name: 'agent-adapter-client', version: '1.0.0'));
  await client.connect(StreamableHttpClientTransport(
    Uri.parse('$url/mcp'),
    opts: StreamableHttpClientTransportOptions(requestInit: {
      'headers': {'Authorization': auth},
    }),
  ));
  final listed = await client.listTools();
  if (!listed.tools.any((tool) => tool.name == 'get_weather')) {
    throw StateError('get_weather missing');
  }
  final result = await client.callTool(
    CallToolRequest(name: 'get_weather', arguments: {'city': 'Paris'}),
  );
  print('MCP: ${(result.content.first as TextContent).text}');
  await client.close();

  final http = HttpClient();
  final request = await http.postUrl(Uri.parse('$url/tools/get_weather/invoke'));
  request.headers.set('Authorization', auth);
  request.headers.contentType = ContentType.json;
  request.write(jsonEncode({
    'arguments': {'city': 'Paris'},
  }));
  final response = await request.close();
  final body = jsonDecode(await response.transform(utf8.decoder).join());
  print('REST: ${body['result']}');
  http.close();
}
