---
search:
  boost: 2
---

# Chat flows by example

`examples/chat_flows.py` is a scripted city help desk (no model, no API key): one agent picks
a `Flow` per question, one per pattern. Run `python examples/chat_flows.py`, or replay it here.

<iframe class="aa-demo" src="../../assets/demo/chat-demo.html" title="Agent Adapter chat: a recorded demo of a city help desk" width="100%" height="760" loading="lazy" sandbox="allow-scripts allow-downloads" allow="clipboard-write"></iframe>

[Open the demo full screen](../assets/demo/chat-demo.html). The Graph lights up node by node.

## The agent

`chat` runs the picked flow in the turn's run, so tool calls show as cards; `session.state`
keeps the museum for later turns. Each flow ends in `reply`, where a stand-in model writes the
answer, so the Cost tab has tokens and dollars.

```python
--8<-- "examples/chat_flows.py:agent"
--8<-- "examples/chat_flows.py:writer"
```

## 1. Router: "Is the City Museum open on Monday?"

`classify` reads the question. A `Route` sends it to one branch: hours, prices or help.

```mermaid
flowchart TD
  aa_start((start)) --> n_classify
  n_classify["classify"]
  n_hours(["hours<br/>tool: opening_hours"])
  n_prices(["prices<br/>tool: ticket_price"])
  n_help["help"]
  n_reply["reply"]
  n_classify --> n_hours
  n_classify --> n_prices
  n_classify -->|"other"| n_help
  n_hours --> n_reply
  n_prices --> n_reply
  n_help --> n_reply
  n_reply --> aa_end((end))
  classDef pause stroke-dasharray: 5 5
```

```python
--8<-- "examples/chat_flows.py:router"
```

## 2. Fan-out, fan-in: "Compare ticket prices at three museums"

`prices` is a `Map`: three `ticket_price` calls run at once. `combine` is a `Join` that waits
for them all, then sorts the prices into a table.

```mermaid
flowchart TD
  aa_start((start)) --> n_plan
  n_plan["plan"]
  n_prices[["prices<br/>fan-out over lookups<br/>tool: ticket_price"]]
  n_combine{{"combine<br/>fan-in: prices"}}
  n_reply["reply"]
  n_plan --> n_prices
  n_prices --> n_combine
  n_combine --> n_reply
  n_reply --> aa_end((end))
  classDef pause stroke-dasharray: 5 5
```

```python
--8<-- "examples/chat_flows.py:fanout"
```

## 3. Approval: "Book two tickets for Saturday"

`book_tickets` has `confirm=True`, so the call waits for Approve or Deny in its card. A
decision is used once. In the recording the demo approves it; a denial fails the turn.

```mermaid
flowchart TD
  aa_start((start)) --> n_details
  n_details["details"]
  n_price(["price<br/>tool: ticket_price"])
  n_book(["book<br/>tool: book_tickets"])
  n_reply["reply"]
  n_details --> n_price
  n_price --> n_book
  n_book --> n_reply
  n_reply --> aa_end((end))
  classDef pause stroke-dasharray: 5 5
```

```python
--8<-- "examples/chat_flows.py:approval"
```

## 4. Loop: "Find a park with a playground near the river"

`search` looks within 0.5 km. `check` doubles the distance when no park has a playground,
and the route goes back to `search`, at most 3 times. Here it widens twice.

```mermaid
flowchart TD
  aa_start((start)) --> n_search
  n_search(["search<br/>tool: find_parks"])
  n_check["check"]
  n_reply["reply"]
  n_search --> n_check
  n_check -.->|"widen (max 3)"| n_search
  n_check -->|"done"| n_reply
  n_reply --> aa_end((end))
  classDef pause stroke-dasharray: 5 5
```

```python
--8<-- "examples/chat_flows.py:loop"
```

## 5. All together: "Plan a rainy day for a family"

The forecast routes to indoor plans. `check` fans out over two museums and `shortlist` joins
them. With fewer than two family picks, it loops back for the next two.

```mermaid
flowchart TD
  aa_start((start)) --> n_forecast
  n_forecast(["forecast<br/>tool: forecast"])
  n_outdoor["outdoor"]
  n_candidates["candidates"]
  n_check[["check<br/>fan-out over lookups<br/>tool: family_check"]]
  n_shortlist{{"shortlist<br/>fan-in: check"}}
  n_reply["reply"]
  n_forecast -->|"rain"| n_candidates
  n_forecast -->|"sun"| n_outdoor
  n_outdoor --> n_reply
  n_candidates --> n_check
  n_check --> n_shortlist
  n_shortlist -.->|"more (max 2)"| n_candidates
  n_shortlist -->|"done"| n_reply
  n_reply --> aa_end((end))
  classDef pause stroke-dasharray: 5 5
```

```python
--8<-- "examples/chat_flows.py:combined"
```

`--check` runs all five questions offline and prints the answers. See also [Chat with your
agent](chat-with-your-agent.md) and [Navigate the graph](navigate-the-graph.md).
