<?php
// Call the tool server from PHP: MCP with the official SDK, then plain REST.
require __DIR__ . '/vendor/autoload.php';

use Mcp\Client;
use Mcp\Client\Transport\HttpTransport;

$url = getenv('AA_URL') ?: 'http://localhost:8000';
$auth = 'Bearer ' . (getenv('AA_TOKEN') ?: 'dev-token');

$client = Client::builder()->setClientInfo('example', '1.0.0')->build();
$client->connect(new HttpTransport("$url/mcp", ['Authorization' => $auth]));
$names = array_map(fn ($tool) => $tool->name, $client->listTools()->tools);
in_array('get_weather', $names, true) or exit("get_weather missing\n");
$result = $client->callTool('get_weather', ['city' => 'Paris']);
echo 'MCP: ' . implode('', array_map(fn ($part) => $part->text, $result->content)) . "\n";
$client->disconnect();

$curl = curl_init("$url/tools/get_weather/invoke");
curl_setopt_array($curl, [
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => json_encode(['arguments' => ['city' => 'Paris']]),
    CURLOPT_HTTPHEADER => ["Authorization: $auth", 'Content-Type: application/json'],
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_FAILONERROR => true,
]);
$body = curl_exec($curl);
$body !== false or exit('REST failed: ' . curl_error($curl) . "\n");
echo 'REST: ' . json_decode($body, true)['result'] . "\n";
