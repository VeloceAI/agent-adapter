// Call the tool server from Rust: MCP with the official SDK, then plain REST.
use anyhow::{anyhow, Result};
use rmcp::{
    model::{CallToolRequestParams, ClientCapabilities, ClientConfig, Implementation, ProtocolVersion},
    transport::{streamable_http_client::StreamableHttpClientTransportConfig, StreamableHttpClientTransport},
    ClientLifecycleMode, ClientServiceExt,
};

#[tokio::main]
async fn main() -> Result<()> {
    let url = std::env::var("AA_URL").unwrap_or("http://localhost:8000".into());
    let token = std::env::var("AA_TOKEN").unwrap_or("dev-token".into());

    let config = StreamableHttpClientTransportConfig::with_uri(format!("{url}/mcp")).auth_header(&token);
    let info = ClientConfig::new(ClientCapabilities::default(), Implementation::new("agent-adapter-client", "1.0.0"));
    let lifecycle = ClientLifecycleMode::Auto {
        preferred_versions: vec![ProtocolVersion::V_2026_07_28],
        legacy_version: Some(ProtocolVersion::V_2025_11_25),
    };
    let client = info.serve_with_lifecycle(StreamableHttpClientTransport::from_config(config), lifecycle).await?;

    let tools = client.list_tools(Default::default()).await?;
    if !tools.tools.iter().any(|tool| tool.name == "get_weather") {
        return Err(anyhow!("get_weather missing"));
    }
    let args = serde_json::json!({ "city": "Paris" });
    let result = client
        .call_tool(CallToolRequestParams::new("get_weather").with_arguments(args.as_object().unwrap().clone()))
        .await?;
    let text = result.content.first().and_then(|c| c.as_text()).ok_or(anyhow!("no text content"))?;
    println!("MCP: {}", text.text);
    client.cancel().await?;

    let reply: serde_json::Value = reqwest::Client::new()
        .post(format!("{url}/tools/get_weather/invoke"))
        .bearer_auth(&token)
        .json(&serde_json::json!({ "arguments": args }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    println!("REST: {}", reply["result"].as_str().ok_or(anyhow!("no result"))?);
    Ok(())
}
