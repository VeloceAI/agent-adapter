// Call the tool server from Go: MCP with the official SDK, then plain REST.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type bearer struct{ token string }

func (b bearer) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.Header.Set("Authorization", "Bearer "+b.token)
	return http.DefaultTransport.RoundTrip(r)
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	url, token := env("AA_URL", "http://localhost:8000"), env("AA_TOKEN", "dev-token")
	ctx := context.Background()
	httpClient := &http.Client{Transport: bearer{token}}

	client := mcp.NewClient(&mcp.Implementation{Name: "agent-adapter-client", Version: "1.0.0"}, nil)
	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: url + "/mcp", HTTPClient: httpClient}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	found := false
	for tool, err := range session.Tools(ctx, nil) {
		if err != nil {
			log.Fatal(err)
		}
		found = found || tool.Name == "get_weather"
	}
	if !found {
		log.Fatal("get_weather missing")
	}
	res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "get_weather", Arguments: map[string]any{"city": "Paris"}})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("MCP:", res.Content[0].(*mcp.TextContent).Text)

	body, _ := json.Marshal(map[string]any{"arguments": map[string]any{"city": "Paris"}})
	resp, err := httpClient.Post(url+"/tools/get_weather/invoke", "application/json", bytes.NewReader(body))
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct{ Result string }
	if resp.StatusCode != 200 || json.NewDecoder(resp.Body).Decode(&out) != nil {
		log.Fatalf("REST failed: HTTP %d", resp.StatusCode)
	}
	fmt.Println("REST:", out.Result)
}
