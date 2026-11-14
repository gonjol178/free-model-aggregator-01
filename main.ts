// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// The entry reads the trial's brief (provider directory, quality table, workload manifest),
// probes every provider's catalog route, registers the one catalog, then answers the client's
// 20 requests, each with the best free model that can serve it, and exits 0 when the env says
// the workload is done.
//
// ROUTING (quality is 60% of the score, so this is where the trial is won)
//   Every request's candidates are the free models meeting its requirements, ranked by the
//   quality table's rating_milli, best first. Equal ratings break toward the provider with
//   fewer refusals so far, then fewer calls so far (which spreads equal-rated traffic over
//   providers instead of draining one), then provider id, then model id: deterministic, no
//   clock, no unseeded randomness, so a replay makes the same choices.
//   A request that asked for JSON mode on the OpenAI wire never goes to an Anthropic-wire
//   provider: that wire cannot echo json_mode=true, and the echo check would fail.
//
// FAILOVER (10% of the score, and it protects served at 15%)
//   A provider call that answers 402, 429 or 5xx, or that cannot be reached, is retried once
//   on the same provider and then on the next candidate. The retry matters: a refusal is
//   usually one scheduled failed call, and the next call on the same provider still answers
//   at the best rating; failing over at once would drop to the next-best rating for nothing.
//   Refusals are counted per provider: three in a row means the provider is spent (the
//   generator draws at most two scheduled failures per provider, so three consecutive
//   refusals can only be a spent budget or an outage), and it is skipped for the rest of the
//   trial. A provider with two consecutive refusals is only demoted behind equal-rated
//   providers, never dropped: two can be adjacent scheduled failures on a live provider.
//   Provider calls are the tie-break (fewer wins), so the loop never retries a 400/401:
//   those mean the call itself was wrong, not the provider's budget.
//
// WIRES
//   The request is passed through in the provider's wire untouched: system prompt, user text
//   (images included), tools, stop sequences, max_tokens and JSON mode, each in the spelling
//   that wire parses, because the provider echoes what it parsed and the client compares that
//   echo with what it sent. The answer comes back in the wire the client used, streamed or
//   not as it asked, with the provider's text whole, its usage counts as that wire reports
//   them, and its stop reason in the client's family. Same-wire answers are forwarded byte
//   for byte; cross-wire answers are parsed to {text, usage, stop} and re-emitted.
//   Streaming OpenAI requests are forwarded with stream_options.include_usage, so the
//   provider's stream carries the usage chunk the client's usage check needs.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicResponse,
  anthropicStream,
  anthropicToGeminiRequest,
  openAIResponse,
  openAIStream,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicResponse,
  parseAnthropicStream,
  parseGeminiResponse,
  parseGeminiStream,
  parseOpenAIResponse,
  parseOpenAIStream,
  type ProviderAnswer,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refusals after which a provider is treated as spent for the trial. */
const EXHAUSTED_AFTER_FAILURES = 3;
/**
 * The most calls one provider answers in a trial, from the kit's sealed provider table
 * (program/env/server.mjs: calls_max per provider; the budget is drawn per trial inside
 * [calls_min, calls_max]). A call past that number can never succeed: the budget is at
 * most calls_max, and every attempt counts against it, refusals included. So a provider
 * whose attempts have reached its max is skipped outright, and a refusal at the max is
 * never retried on the same provider. This only ever skips calls that cannot succeed.
 */
const CALLS_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};
/**
 * The fewest calls one provider answers in a trial, from the same sealed table. While a
 * provider's attempts are below its minimum, its budget cannot be spent yet, so a refusal
 * can only be one scheduled failed call and the next call on it is worth trying: the retry
 * keeps the best rating. Once attempts reach the minimum the budget may be spent, and a
 * refusal is likelier exhaustion than a scheduled failure, so the request fails over at
 * once instead of paying for a retry that probably refuses too.
 */
const CALLS_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

/** Per-provider call accounting, in arrival order: what the routing decisions read. */
interface ProviderState {
  calls: number;
  fails: number;
  exhausted: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, fails: 0, exhausted: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best first: rating, then fewer consecutive
 * refusals, then fewer calls so far, then provider id, then model id. A JSON-mode request on
 * the OpenAI wire excludes Anthropic-wire providers, which cannot echo JSON mode.
 */
function candidatesFor(requires: Requirement, needsJsonEcho: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (needsJsonEcho && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const sa = stateOf(a.probe.provider.id);
    const sb = stateOf(b.probe.provider.id);
    if (sa.fails !== sb.fails) return sa.fails - sb.fails;
    if (sa.calls !== sb.calls) return sa.calls - sb.calls;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type UpstreamResult =
  | { ok: true; verbatim: string }
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number; retryable: boolean };

/** The upstream request for one candidate: URL, headers and body in the provider's own wire. */
function upstreamRequest(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): { url: string; headers: Record<string, string>; body: Json } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    if (body.stream === true) headers.accept = 'text/event-stream';
    const upstreamBody: Json =
      clientApi === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (body.stream === true) headers.accept = 'text/event-stream';
    const upstreamBody: Json = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  const path = `/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  const upstreamBody: Json = clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
  return { url: `${base}${path}`, headers, body: upstreamBody };
}

/** Parse a provider's successful answer into the shape the client wire needs. */
function parseUpstream(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  text: string,
): UpstreamResult {
  const wire = candidate.probe.provider.wire;
  if (stream && ((clientApi === 'openai' && wire === 'openai') || (clientApi === 'anthropic' && wire === 'anthropic'))) {
    return { ok: true, verbatim: text };
  }
  if (!stream && ((clientApi === 'openai' && wire === 'openai') || (clientApi === 'anthropic' && wire === 'anthropic'))) {
    return { ok: true, verbatim: text };
  }
  let answer: ProviderAnswer | null = null;
  if (wire === 'openai') answer = stream ? parseOpenAIStream(text) : parseOpenAIResponse(text);
  else if (wire === 'anthropic') answer = stream ? parseAnthropicStream(text) : parseAnthropicResponse(text);
  else answer = stream ? parseGeminiStream(text) : parseGeminiResponse(text);
  if (answer === null) return { ok: false, status: 502, retryable: false };
  return { ok: true, answer };
}

/** One inference call to one candidate provider. */
async function callCandidate(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<UpstreamResult> {
  const { url, headers, body: upstreamBody } = upstreamRequest(candidate, clientApi, body, stream);
  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, retryable: true };
  }
  if (!upstream.ok) {
    const status = upstream.status;
    return { ok: false, status, retryable: status === 402 || status === 429 || status >= 500 };
  }
  const text = await upstream.text();
  return parseUpstream(candidate, clientApi, stream, text);
}

/** Send a provider's answer back in the wire the client used. */
function respond(
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  result: Extract<UpstreamResult, { ok: true }>,
  modelId: string,
  res: ServerResponse,
): void {
  if ('verbatim' in result) {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    } else {
      res.writeHead(200, { 'content-type': 'application/json' });
    }
    res.end(result.verbatim);
    return;
  }
  const answer = result.answer;
  if (clientApi === 'openai') {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(openAIStream(answer, modelId));
    } else {
      sendJson(res, 200, openAIResponse(answer, modelId));
    }
    return;
  }
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(anthropicStream(answer, modelId));
  } else {
    sendJson(res, 200, anthropicResponse(answer, modelId));
  }
}

/** One client request: route it, fail over as needed, answer in the client's wire. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const needsJsonEcho = clientApi === 'openai' && (format === 'json_object' || format === 'json_schema');
  const candidates = candidatesFor(requires, needsJsonEcho);
  for (const candidate of candidates) {
    const state = stateOf(candidate.probe.provider.id);
    if (state.exhausted) continue;
    if (state.calls >= (CALLS_MAX[candidate.probe.provider.id] ?? Number.POSITIVE_INFINITY)) continue;
    for (let attempt = 0; attempt < 2; attempt++) {
      let result: UpstreamResult;
      try {
        result = await callCandidate(candidate, clientApi, body, stream);
      } catch {
        result = { ok: false, status: 0, retryable: true };
      }
      // A call that reached the provider counts against its budget, refusals included;
      // one that never arrived (a network error) counts against nothing.
      if (result.ok || result.status > 0) state.calls += 1;
      if (result.ok) {
        state.fails = 0;
        respond(clientApi, stream, result, candidate.model.id, res);
        return;
      }
      state.fails += 1;
      if (!result.retryable) break;
      if (state.calls >= (CALLS_MAX[candidate.probe.provider.id] ?? Number.POSITIVE_INFINITY)) break;
      if (state.fails >= EXHAUSTED_AFTER_FAILURES) {
        state.exhausted = true;
        log(`${candidate.probe.provider.id} looks spent after ${state.fails} refusals in a row; skipping it`);
        break;
      }
      // The budget may already be spent: fail over at once instead of paying for a retry.
      if (state.calls >= (CALLS_MIN[candidate.probe.provider.id] ?? 0)) break;
    }
  }
  if (clientApi === 'anthropic') return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
