/** M8: the client never follows a redirect, so the bearer token and body never reach another URL. */
import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { IncomingHttpHeaders, RequestListener, Server } from "node:http";
import type { AddressInfo } from "node:net";
import { after, before, describe, test } from "node:test";

import { HttpClient } from "../src/http.ts";
import { ToolClient, ToolTransportError } from "../src/index.ts";
import { json, rejection } from "./helpers.ts";

async function listen(handler: RequestListener): Promise<{ server: Server; url: string }> {
  const server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return { server, url: `http://127.0.0.1:${port}` };
}

async function close(server: Server): Promise<void> {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
}

describe("redirects", () => {
  test("every request asks fetch to refuse redirects", async () => {
    const seen: (RequestRedirect | undefined)[] = [];
    const fetch = (async (_input: string | URL | Request, init: RequestInit = {}) => {
      seen.push(init.redirect);
      return json(200, {});
    }) as typeof globalThis.fetch;
    const client = new HttpClient({ baseUrl: "http://srv", fetch, token: "secret" });
    await client.request({ method: "GET", path: "/a" });
    await client.request({ method: "POST", path: "/b", body: {} });
    (await client.open({ method: "GET", path: "/c" })).close();
    assert.deepEqual(seen, ["error", "error", "error"]);
  });

  describe("against a real server", () => {
    const leaked: IncomingHttpHeaders[] = [];
    let target: { server: Server; url: string };
    let origin: { server: Server; url: string };

    before(async () => {
      target = await listen((req, res) => {
        leaked.push(req.headers);
        res.writeHead(200, { "content-type": "application/json" }).end("[]");
      });
      origin = await listen((req, res) => {
        res.writeHead(307, { location: `${target.url}${req.url ?? "/"}` }).end();
      });
    });

    after(async () => {
      await close(origin.server);
      await close(target.server);
    });

    for (const method of ["GET", "POST"] as const) {
      test(`a ${method} redirect is a transport error and the token stays home`, async () => {
        const client = new HttpClient({ baseUrl: origin.url, token: "secret", retry: false });
        const request = method === "GET" ? { method, path: "/tools" } : { method, path: "/x", body: {} };
        const err = await rejection(client.request(request));
        assert.ok(err instanceof ToolTransportError);
        assert.equal(err.reason, "network");
        assert.deepEqual(leaked, []);
      });
    }

    test("ToolClient refuses a redirected call too", async () => {
      const client = new ToolClient({ baseUrl: origin.url, token: "secret", retry: false });
      await assert.rejects(client.listTools(), ToolTransportError);
      assert.deepEqual(leaked, []);
    });
  });
});
