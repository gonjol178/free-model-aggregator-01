// Cross-wire translations the starter does not implement.
//
// The trial's client speaks OpenAI or Anthropic; a provider speaks OpenAI, Anthropic or Gemini.
// Every function here is a pure function of its arguments: no clock, no randomness, so a replay
// sees the same bytes. The one invariant that matters most: the provider echoes what it parsed
// (system prompt, user text, tools, max_tokens, stops, image, JSON mode), and the client compares
// that echo with what it sent, so a translation must carry every one of those fields through
// unchanged, in the provider wire's own spelling.
//
// Wire limits that shape routing (callers must enforce them):
//   - an Anthropic-wire provider can never echo json_mode=true (its parser hard-codes false), so
//     an OpenAI client request that asked for JSON mode must never go to one;
//   - a Gemini-wire provider carries JSON mode only via generationConfig.responseMimeType.
// The simulated providers always answer plain text with a family-0 stop reason, so no response
// ever carries tool calls; the translations still map tool calls where the shape allows them.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

const FIXED_CREATED = 1767225600;
/** The provider's SSE text chunks are 13 characters; re-emitted streams use the same cut. */
const CHUNK_CHARS = 13;

/** A provider's answer, reduced to what every client wire needs. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop token: "stop" | "end_turn" | "STOP" (family 0 in this trial). */
  stop: string;
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function splitImageUrl(url: string): { mime: string; data: string } | { uri: string } | null {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (m !== null) return { mime: m[1] ?? 'image/png', data: m[2] ?? '' };
  if (url !== '') return { uri: url };
  return null;
}

// ---------------------------------------------------------------------------
// Requests: OpenAI client -> Anthropic provider
// ---------------------------------------------------------------------------

/** One OpenAI content part as an Anthropic content block. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const img = splitImageUrl(part.image_url.url);
    if (img === null) return null;
    if ('data' in img) return { type: 'image', source: { type: 'base64', media_type: img.mime, data: img.data } };
    return { type: 'image', source: { type: 'url', url: img.uri } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`.
 * JSON mode is deliberately NOT carried: the Anthropic wire cannot echo it, so callers must
 * never route a JSON-mode request here.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input,
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToAnthropic(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Requests: OpenAI or Anthropic client -> Gemini provider
// ---------------------------------------------------------------------------

function geminiImagePart(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

function geminiPartsFromOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const img = splitImageUrl(part.image_url.url);
      if (img === null) continue;
      if ('data' in img) parts.push({ inlineData: { mimeType: img.mime, data: img.data } });
      else parts.push({ fileData: { fileUri: img.uri } });
    }
  }
  return parts;
}

function geminiPartsFromAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const part = geminiImagePart(block.source);
      if (part !== null) parts.push(part);
    }
    // tool_result blocks are results, not user text: skipping them keeps the user text exact.
  }
  return parts;
}

function geminiFunctionDeclarations(tools: unknown, fromOpenAI: boolean): Json[] {
  if (!Array.isArray(tools)) return [];
  const out: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw)) continue;
    if (fromOpenAI) {
      if (!isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      out.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    } else {
      if (typeof raw.name !== 'string') continue;
      out.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return out;
}

function geminiGenerationConfig(opts: {
  maxTokens?: number;
  stops?: string[];
  temperature?: number;
  topP?: number;
  jsonMode?: boolean;
}): Json {
  const config: Json = {};
  if (opts.maxTokens !== undefined) config.maxOutputTokens = opts.maxTokens;
  if (opts.stops !== undefined && opts.stops.length > 0) config.stopSequences = opts.stops;
  if (opts.temperature !== undefined) config.temperature = opts.temperature;
  if (opts.topP !== undefined) config.topP = opts.topP;
  if (opts.jsonMode === true) config.responseMimeType = 'application/json';
  return config;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body (model goes in the URL). */
export function openAIToGeminiRequest(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts = geminiPartsFromOpenAI(raw.content);
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') continue; // a tool result is not user text; skipping keeps the echo exact
    if (raw.role !== 'user') continue;
    const parts = geminiPartsFromOpenAI(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = {};
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  out.contents = contents;
  const decls = geminiFunctionDeclarations(body.tools, true);
  if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : undefined;
  out.generationConfig = geminiGenerationConfig({
    maxTokens,
    stops,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    jsonMode: format === 'json_object' || format === 'json_schema',
  });
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body (model goes in the URL). */
export function anthropicToGeminiRequest(body: Json): Json {
  const systemText = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts = geminiPartsFromAnthropic(raw.content);
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts = geminiPartsFromAnthropic(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = {};
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  out.contents = contents;
  const decls = geminiFunctionDeclarations(body.tools, false);
  if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s) => typeof s === 'string') : undefined;
  out.generationConfig = geminiGenerationConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
    stops,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    jsonMode: false, // the Anthropic wire never asks for JSON mode
  });
  return out;
}

// ---------------------------------------------------------------------------
// Responses: any provider wire -> ProviderAnswer
// ---------------------------------------------------------------------------

/** The stop family of a provider stop token: 0 normal, 1 length, 2 tool, 3 refusal. */
function stopFamily(stop: string): number {
  if (stop === 'stop' || stop === 'end_turn' || stop === 'STOP') return 0;
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 1;
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 2;
  return 3;
}

export function toOpenAIStop(stop: string): string {
  const f = stopFamily(stop);
  if (f === 1) return 'length';
  if (f === 2) return 'tool_calls';
  if (f === 3) return 'content_filter';
  return 'stop';
}

export function toAnthropicStop(stop: string): string {
  const f = stopFamily(stop);
  if (f === 1) return 'max_tokens';
  if (f === 2) return 'tool_use';
  if (f === 3) return 'refusal';
  return 'end_turn';
}

/** A Chat Completions JSON body as an answer. */
export function parseOpenAIResponse(text: string): ProviderAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** An Anthropic Messages JSON body as an answer. */
export function parseAnthropicResponse(text: string): ProviderAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(value) || !Array.isArray(value.content)) return null;
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: textOf(value.content),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
  };
}

/** A Gemini generateContent JSON body as an answer. */
export function parseGeminiResponse(text: string): ProviderAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(value) || !Array.isArray(value.candidates) || value.candidates.length === 0) return null;
  const candidate = isRecord(value.candidates[0]) ? value.candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const textOut = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: textOut,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
  };
}

interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames: `{event, data}` per blank-line-separated block. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI SSE stream as an answer. */
export function parseOpenAIStream(text: string): ProviderAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = jsonOrNull(frame.data);
    if (chunk === null) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') out += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return sawAny ? { text: out, inputTokens: input, outputTokens: output, stop } : null;
}

/** An Anthropic SSE stream as an answer. */
export function parseAnthropicStream(text: string): ProviderAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    if (frame.event === null) continue;
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return sawAny ? { text: out, inputTokens: input, outputTokens: output, stop } : null;
}

/** A Gemini SSE stream as an answer. */
export function parseGeminiStream(text: string): ProviderAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    sawAny = true;
    if (Array.isArray(data.candidates) && data.candidates.length > 0) {
      const candidate = isRecord(data.candidates[0]) ? data.candidates[0] : null;
      if (candidate !== null) {
        const content = isRecord(candidate.content) ? candidate.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const p of parts) if (isRecord(p) && typeof p.text === 'string') out += p.text;
        if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
      }
    }
    if (isRecord(data.usageMetadata)) {
      input = numberOrZero(data.usageMetadata.promptTokenCount);
      output = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return sawAny ? { text: out, inputTokens: input, outputTokens: output, stop } : null;
}

// ---------------------------------------------------------------------------
// Responses: ProviderAnswer -> a client wire
// ---------------------------------------------------------------------------

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += CHUNK_CHARS) pieces.push(text.slice(at, at + CHUNK_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** An answer as a Chat Completions JSON body naming `model`. */
export function openAIResponse(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: toOpenAIStop(answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** An answer as an Anthropic Messages JSON body naming `model`. */
export function anthropicResponse(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: toAnthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** An answer as a Chat Completions SSE stream naming `model`, usage chunk included. */
export function openAIStream(answer: ProviderAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, toOpenAIStop(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An answer as an Anthropic SSE stream naming `model`. */
export function anthropicStream(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: toAnthropicStop(answer.stop), stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
