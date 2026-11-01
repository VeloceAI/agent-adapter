// For the Python tests: `node graph_cli.mjs mermaid GRAPH.json` or `snapshot DETAIL.json FLOW`.
import { readFileSync } from "node:fs";

import { load, plain } from "./load.mjs";

const AA = load("graphlive.js");
const [command, path, flow] = process.argv.slice(2);
const data = JSON.parse(readFileSync(path, "utf8"));

if (command === "mermaid") {
  process.stdout.write(AA.graphLive.mermaid(data));
} else if (command === "snapshot") {
  const graph = data.graphs[flow];
  const snap = AA.graphLive.snapshot(graph, data);
  const steps = AA.graphLive.replaySteps(data.events, flow);
  process.stdout.write(JSON.stringify(plain({ nodes: snap.nodes, edges: snap.edges, steps: steps.length, mermaid: AA.graphLive.mermaid(graph) })));
} else {
  process.stderr.write("usage: graph_cli.mjs mermaid GRAPH.json | snapshot DETAIL.json FLOW\n");
  process.exit(2);
}
