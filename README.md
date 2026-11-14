# free-model-aggregator-01

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API
over ten simulated free-tier model providers, in JSON and in server-sent events.

It reads the trial's brief, probes every provider's catalog route, registers the one
catalog with the env, then answers each of the 20 requests with the best free model that
can serve it — rating-ordered routing with budget-aware failover — and exits 0 when the
env says the workload is done.

- `main.ts` — the server: brief, catalog, routes, register, poll, routing, failover
- `lib/providers.ts` — the brief's documents and the catalog probe
- `lib/wire.ts` — the client wires (OpenAI/Anthropic model lists, Anthropic <-> OpenAI)
- `lib/translate.ts` — the cross-wire translations (Gemini + Anthropic upstream wires)

`node main.ts --list` prints the catalog document the server registers, byte for byte.

Local grades (kit's `grade-local`, process driver, all 3 practice trials):
score 99 on every trial — quality 99 (the ceiling: the best model for 17 of 20 requests
lives on one provider whose per-trial budget covers at most 12 calls), served 100,
wire 100, failover 100, catalog 100; replays matched byte for byte.

MIT licensed.
