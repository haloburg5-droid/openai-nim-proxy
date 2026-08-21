// server.js - OpenAI-compatible proxy for the NVIDIA NIM API
//
// Works with BOTH SillyTavern and JanitorAI:
//   - SillyTavern sends a rich sampler set. Anything NIM understands is forwarded untouched.
//   - JanitorAI sends almost nothing. Missing values fall back to the DEFAULTS below.
//
// Design notes:
//   - ALLOWLIST, not passthrough. NIM rejects unknown fields, and SillyTavern's own docs warn
//     that custom endpoints "may error upon invalid request", so we only forward known-good keys.
//   - Params NIM may or may not accept (top_k, repetition_penalty, min_p...) are sent in an
//     "extended" bucket. If NIM 400/422s, we strip that bucket and retry once automatically.
//   - Uses ?? (nullish) not ||, so an explicit 0 from the client is honoured rather than
//     silently replaced by a default.

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '20mb' }));

// ---------------------------------------------------------------------------
// Config (all overridable with environment variables)
// ---------------------------------------------------------------------------
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

// Show the model's reasoning wrapped in <think> tags.
// false = strip it. SillyTavern can parse/fold <think> itself, so set SHOW_REASONING=true
// there if you want to see it; JanitorAI folds it too but is happier without.
const SHOW_REASONING = process.env.SHOW_REASONING === 'true';

// Ask GLM/Qwen to use their native thinking mode. Off by default: combined with a
// heavy system prompt it makes GLM loop inside <think>.
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';

// JanitorAI sends 0 for sliders the user never touched, which is indistinguishable from
// a deliberate 0. When true, a 0 for the penalty/top_k family is treated as "unset" so the
// DEFAULTS below apply. SillyTavern users who genuinely want 0 should set this to false.
const ZERO_MEANS_UNSET = process.env.ZERO_MEANS_UNSET !== 'false';

// Log each outgoing request's sampler set. Handy while tuning.
const DEBUG = process.env.DEBUG === 'true';

// NVIDIA takes models offline without warning ("DEGRADED function cannot be invoked")
// and not every catalogue model is callable on every account ("not found for account").
// When that happens, retry on the next model in this list instead of erroring.
//
// IMPORTANT: only list models strong enough for your prompt. A heavy RP preset (long
// system prompt + chain-of-thought + strict formatting) will make a small model degenerate
// into repeated-token garbage like "[[[[[[[[". That is why no 8B model is listed here.
// Set FALLBACK_MODELS="" to disable fallback entirely and get a clear error instead.
const FALLBACK_MODELS = (
  process.env.FALLBACK_MODELS ??
  'deepseek-ai/deepseek-v4-pro,meta/llama-3.1-70b-instruct'
).split(',').map((s) => s.trim()).filter(Boolean);

// max_tokens is a hard guillotine, not a target: generation stops the moment it is hit,
// mid-sentence if necessary. It must cover the WHOLE completion, so a preset asking for
// 400-600 words (~800 tokens) that ALSO runs a chain-of-thought block needs budget for both.
//
// MAX_TOKENS env var:
//   unset / 'none' / 'unlimited' / '0'  -> send NO max_tokens at all. The model writes until
//                                          it finishes or hits the context limit. This is the
//                                          default, and matches "no limit".
//   a number (e.g. 2048)                -> use it whenever the client does not send its own.
//
// A client value always wins. SillyTavern sends its "Response (tokens)" slider, so set it
// there. JanitorAI sends 0 or nothing, which is what this default covers.
//
// Trade-off of unlimited: a runaway reply can exceed a gateway timeout (Vercel/Cloudflare cut
// at ~100s). Keep streaming ON to mitigate that, and if 524s appear, set MAX_TOKENS=2048.
const MAX_TOKENS_RAW = String(process.env.MAX_TOKENS ?? '').trim().toLowerCase();
const MAX_TOKENS_DEFAULT =
  (MAX_TOKENS_RAW === '' || MAX_TOKENS_RAW === '0' || MAX_TOKENS_RAW === 'none' || MAX_TOKENS_RAW === 'unlimited')
    ? null                              // null = omit the field entirely = no limit
    : (Number(MAX_TOKENS_RAW) || null);

const DEFAULTS = {
  temperature:       0.8,   // GLM 5.2 community sweet spot
  top_p:             0.95,  // GLM default; avoid tuning this and temperature together
  frequency_penalty: 0.2,   // mild anti-repetition; higher values wreck long replies
  presence_penalty:  0.2
};

// Sampler keys that are only meaningful if the client actually sent them.
const ZEROABLE = new Set([
  'frequency_penalty', 'presence_penalty', 'top_k', 'repetition_penalty', 'min_p', 'seed'
]);

// Forwarded as-is. Core OpenAI schema, universally accepted by NIM.
const CORE_PARAMS = [
  'temperature', 'top_p', 'max_tokens', 'stop', 'frequency_penalty',
  'presence_penalty', 'seed', 'n', 'logprobs', 'top_logprobs', 'response_format'
];

// Accepted by most but not all NIM models. Stripped and retried on a 400/422.
const EXTENDED_PARAMS = ['top_k', 'repetition_penalty', 'min_p', 'ignore_eos', 'min_tokens'];

// Some slider values are valid to the API but catastrophic for roleplay, because they
// REWARD repetition instead of discouraging it. A model given these will emit one token
// forever ("[[[[[[[", "XXXXXX"):
//   frequency_penalty / presence_penalty < 0  -> negative penalty = bonus for repeats
//   repetition_penalty < 1                    -> below 1.0 rewards repeats (1.0 = neutral)
// Clamp them to the neutral floor and log it. Set CLAMP_SAMPLERS=false to allow raw values.
const CLAMP_SAMPLERS = process.env.CLAMP_SAMPLERS !== 'false';
const SAMPLER_FLOORS = { frequency_penalty: 0, presence_penalty: 0, repetition_penalty: 1 };

function clampSamplers(payload) {
  if (!CLAMP_SAMPLERS) return payload;
  for (const [key, floor] of Object.entries(SAMPLER_FLOORS)) {
    if (payload[key] !== undefined && payload[key] < floor) {
      console.warn(`[proxy] clamped ${key} ${payload[key]} -> ${floor} (values below ${floor} cause repetition loops)`);
      payload[key] = floor;
    }
  }
  return payload;
}

// Samplers SillyTavern offers that NIM does not implement. Dropped silently so they
// never trigger a 400. (ST's docs note these have no effect on stricter endpoints.)
const UNSUPPORTED = [
  'top_a', 'typical_p', 'tfs', 'epsilon_cutoff', 'eta_cutoff', 'mirostat',
  'mirostat_tau', 'mirostat_eta', 'smoothing_factor', 'smoothing_curve',
  'dynatemp', 'dynatemp_low', 'dynatemp_high', 'encoder_repetition_penalty',
  'no_repeat_ngram_size', 'penalty_alpha', 'guidance_scale', 'repetition_penalty_range'
];

const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'thinkingmachines/inkling',
  'gpt-4':          'moonshotai/kimi-k3',
  'gpt-4-turbo':    'openai/gpt-oss-120b',
  'gpt-4o':         'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-opus':  'z-ai/glm-5.2',
  'claude-3-sonnet':'meta/muse-glimmer-30b',
  'gemini-pro':     'deepseek-ai/deepseek-v4-flash-0731'
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Treat null/undefined as unset. Optionally treat 0 as unset for slider-style params.
function isUnset(key, value) {
  if (value === undefined || value === null) return true;
  if (ZERO_MEANS_UNSET && value === 0 && ZEROABLE.has(key)) return true;
  return false;
}

// Remember models we have already confirmed exist, so we probe only once each.
const verifiedModels = new Set();

async function resolveModel(requestedModel) {
  if (!requestedModel) return 'meta/llama-3.1-8b-instruct';

  // Explicit alias from the table above.
  if (MODEL_MAPPING[requestedModel]) return MODEL_MAPPING[requestedModel];

  // Already checked this one.
  if (verifiedModels.has(requestedModel)) return requestedModel;

  // A vendor/model style name (what you type in SillyTavern) is almost certainly a real
  // NIM id, so use it directly instead of burning a probe request.
  if (requestedModel.includes('/')) {
    verifiedModels.add(requestedModel);
    return requestedModel;
  }

  // Unknown bare name: probe NIM once with a 1-token request.
  try {
    const probe = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      { model: requestedModel, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
      {
        headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        validateStatus: (s) => s < 500,
        timeout: 15000
      }
    );
    if (probe.status >= 200 && probe.status < 300) {
      verifiedModels.add(requestedModel);
      return requestedModel;
    }
  } catch (_) { /* fall through to heuristic */ }

  const m = requestedModel.toLowerCase();
  if (m.includes('gpt-4') || m.includes('opus') || m.includes('405b')) return 'meta/llama-3.1-405b-instruct';
  if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) return 'meta/llama-3.1-70b-instruct';
  return 'meta/llama-3.1-8b-instruct';
}

// Turn an incoming client request into a NIM payload.
function buildPayload(body, nimModel, { includeExtended = true } = {}) {
  const payload = {
    model: nimModel,
    messages: body.messages,
    stream: body.stream === true
  };

  // Core params: client value if present, otherwise our default (if we have one).
  for (const key of CORE_PARAMS) {
    if (!isUnset(key, body[key])) {
      payload[key] = body[key];
    } else if (DEFAULTS[key] !== undefined) {
      payload[key] = DEFAULTS[key];
    }
  }

  // A client sending 0 (or nothing) means "no preference". Fall back to MAX_TOKENS if one
  // is configured, otherwise omit the field entirely so the model is uncapped.
  if (!payload.max_tokens || payload.max_tokens <= 0) {
    if (MAX_TOKENS_DEFAULT) payload.max_tokens = MAX_TOKENS_DEFAULT;
    else delete payload.max_tokens;
  }

  // SillyTavern sends stop sequences as `stop` or `stop_sequence`; normalise and cap at 4.
  const stops = body.stop ?? body.stop_sequence ?? body.stopping_strings;
  if (Array.isArray(stops) && stops.length) {
    payload.stop = stops.filter(Boolean).slice(0, 4);
  } else if (typeof stops === 'string' && stops.trim()) {
    payload.stop = [stops];
  }

  // Extended params only if the client explicitly asked for them.
  if (includeExtended) {
    for (const key of EXTENDED_PARAMS) {
      if (!isUnset(key, body[key])) payload[key] = body[key];
    }
  }

  // Native thinking mode for models that support the flag.
  if (ENABLE_THINKING_MODE) {
    const lower = nimModel.toLowerCase();
    const kwargs = {};
    if (lower.includes('qwen') || lower.includes('glm')) kwargs.thinking = true;
    if (Object.keys(kwargs).length) payload.chat_template_kwargs = kwargs;
  }

  return clampSamplers(payload);
}

// An axios error body is a stream when responseType was 'stream', so read it back to text.
async function readErrorBody(error) {
  const data = error.response?.data;
  if (!data) return null;
  if (typeof data === 'string') return data;
  if (typeof data.pipe === 'function') {
    try {
      const chunks = [];
      for await (const c of data) chunks.push(c);
      return Buffer.concat(chunks).toString();
    } catch (_) { return null; }
  }
  return JSON.stringify(data);
}

// --- Degeneration guard -----------------------------------------------------
// A model can collapse into emitting one tiny token forever ("[[[[[[[", "XXXXXX").
// Once that starts it never recovers, and if the junk lands in the chat history it
// poisons every later turn. So we cut it off rather than relay thousands of tokens.
const DEGEN_REPEATS = Number(process.env.DEGEN_REPEATS) || 30;

// Trailing run of one repeated short fragment? Return where it starts, else -1.
function findDegenerateTail(text) {
  if (!text || text.length < DEGEN_REPEATS) return -1;
  for (let unit = 1; unit <= 3; unit++) {
    const tail = text.slice(-unit);
    if (!tail.trim()) continue;              // ignore whitespace runs
    let count = 0;
    let i = text.length;
    while (i - unit >= 0 && text.slice(i - unit, i) === tail) { count++; i -= unit; }
    if (count >= DEGEN_REPEATS) return i;
  }
  return -1;
}

// True when the model itself is the problem, so a different model would likely work.
function isModelUnavailable(status, bodyText) {
  if (status !== 400 && status !== 404) return false;
  const t = (bodyText || '').toLowerCase();
  return t.includes('degraded') || t.includes('not found for account') || t.includes('does not exist');
}

async function callNim(payload, wantStream) {
  return axios.post(`${NIM_API_BASE}/chat/completions`, payload, {
    headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
    responseType: wantStream ? 'stream' : 'json',
    timeout: wantStream ? 0 : 120000
  });
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------
app.get(['/health', '/v1/health'], (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    api_key_configured: Boolean(NIM_API_KEY),
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    zero_means_unset: ZERO_MEANS_UNSET,
    defaults: DEFAULTS
  });
});

// Diagnostic: ask NVIDIA which models THIS account can actually call.
// Open in a browser to get the authoritative list. Add ?q=glm to filter.
// Use this whenever you get "Function ... not found for account", which means the
// model id you asked for is not callable with your key.
app.get('/nim/models', async (req, res) => {
  try {
    const upstream = await axios.get(`${NIM_API_BASE}/models`, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}` },
      timeout: 20000
    });
    let ids = (upstream.data?.data || []).map((m) => m.id).sort();
    if (req.query.q) {
      const q = String(req.query.q).toLowerCase();
      ids = ids.filter((id) => id.toLowerCase().includes(q));
    }
    res.json({ count: ids.length, models: ids });
  } catch (error) {
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.error?.message || error.message,
        hint: 'If this fails, the NIM_API_KEY is wrong or lacks access.'
      }
    });
  }
});

app.get(['/v1/models', '/models'], (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map((id) => ({
      id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim-proxy'
    }))
  });
});

// ---------------------------------------------------------------------------
// DIAGNOSTIC ECHO. Point your client's proxy URL at /echo/v1/chat/completions and
// send a message: instead of calling NVIDIA, the reply IS a dump of exactly what
// arrived here. Use it to see what a client (or a middleman like LoreBary) really
// sends, and to compare two clients against each other.
// ---------------------------------------------------------------------------
app.post(['/echo/v1/chat/completions', '/echo/chat/completions'], (req, res) => {
  const b = req.body || {};
  const msgs = Array.isArray(b.messages) ? b.messages : [];

  const lines = [];
  lines.push('=== ECHO: what this proxy received ===');
  lines.push(`model: ${JSON.stringify(b.model)}   stream: ${JSON.stringify(b.stream)}`);
  lines.push(`content-type: ${req.headers['content-type'] || 'none'}`);
  lines.push(`user-agent: ${(req.headers['user-agent'] || 'none').slice(0, 80)}`);

  const samplers = ['temperature','top_p','top_k','max_tokens','frequency_penalty',
                    'presence_penalty','repetition_penalty','min_p','seed','stop'];
  lines.push('samplers: ' + samplers
    .filter((k) => b[k] !== undefined)
    .map((k) => `${k}=${JSON.stringify(b[k])}`).join(', ') || 'samplers: (none sent)');

  const unknown = Object.keys(b).filter(
    (k) => !['model','messages','stream','prompt', ...samplers].includes(k)
  );
  lines.push(`other keys: ${unknown.length ? unknown.join(', ') : '(none)'}`);

  lines.push('');
  lines.push(`messages: ${msgs.length}`);
  const total = msgs.reduce((n, m) => n + String(m?.content ?? '').length, 0);
  lines.push(`total content chars: ${total} (~${Math.round(total / 4)} tokens)`);
  lines.push('');

  msgs.forEach((m, i) => {
    const c = typeof m?.content === 'string' ? m.content : JSON.stringify(m?.content);
    const text = String(c ?? '');
    lines.push(`--- [${i}] role=${JSON.stringify(m?.role)} len=${text.length} ---`);
    lines.push(text.length > 300 ? text.slice(0, 200) + '\n   ...[trimmed]...\n' + text.slice(-100) : text);
  });

  // Flag the things that most often break a model.
  lines.push('');
  lines.push('=== sanity checks ===');
  lines.push(`empty/blank message contents : ${msgs.filter((m) => !String(m?.content ?? '').trim()).length}`);
  lines.push(`non-string contents          : ${msgs.filter((m) => typeof m?.content !== 'string').length}`);
  lines.push(`roles in order               : ${msgs.map((m) => m?.role).join(' > ') || '(none)'}`);
  lines.push(`last role                    : ${msgs.length ? JSON.stringify(msgs[msgs.length - 1].role) : '(none)'}`);
  const looksJson = msgs.some((m) => /^\s*[[{]\s*"?(role|messages)"?\s*[:[]/.test(String(m?.content ?? '')));
  lines.push(`content looks double-encoded : ${looksJson}`);

  const dump = lines.join('\n');
  console.log('[echo]\n' + dump);

  if (b.stream === true) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('X-Accel-Buffering', 'no');
    res.write(`data: ${JSON.stringify({
      id: 'echo', object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000),
      model: 'echo', choices: [{ index: 0, delta: { role: 'assistant', content: dump }, finish_reason: null }]
    })}\n\n`);
    res.write(`data: ${JSON.stringify({
      id: 'echo', object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000),
      model: 'echo', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
    })}\n\n`);
    res.write('data: [DONE]\n\n');
    return res.end();
  }

  res.json({
    id: 'echo', object: 'chat.completion', created: Math.floor(Date.now() / 1000), model: 'echo',
    choices: [{ index: 0, message: { role: 'assistant', content: dump }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  });
});

app.post(['/v1/chat/completions', '/chat/completions'], async (req, res) => {
  const wantStream = req.body?.stream === true;

  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: { message: 'NIM_API_KEY is not set on the server.', type: 'invalid_request_error', code: 500 }
      });
    }
    if (!Array.isArray(req.body?.messages) || req.body.messages.length === 0) {
      return res.status(400).json({
        error: { message: 'messages array is required.', type: 'invalid_request_error', code: 400 }
      });
    }

    const nimModel = await resolveModel(req.body.model);
    let payload = buildPayload(req.body, nimModel);

    if (DEBUG) {
      const { messages, ...rest } = payload;
      console.log('[proxy] ->', JSON.stringify(rest), `(${messages.length} messages)`);
      const dropped = UNSUPPORTED.filter((k) => req.body[k] !== undefined);
      if (dropped.length) console.log('[proxy] dropped unsupported:', dropped.join(', '));
    }

    // Try the requested model, then each fallback, if the model itself is unavailable.
    const candidates = [nimModel, ...FALLBACK_MODELS.filter((m) => m !== nimModel)];
    let nimResponse = null;
    let lastError = null;
    let servedBy = nimModel;

    for (const candidate of candidates) {
      payload = buildPayload(req.body, candidate);
      try {
        nimResponse = await callNim(payload, wantStream);
        servedBy = candidate;
        break;
      } catch (err) {
        const status = err.response?.status;
        const bodyText = await readErrorBody(err);

        // A rejected sampler is fixable on this same model: strip extras and retry once.
        const hadExtended = EXTENDED_PARAMS.some((k) => payload[k] !== undefined);
        if ((status === 400 || status === 422) && hadExtended && !isModelUnavailable(status, bodyText)) {
          console.warn(`[proxy] ${candidate}: retrying without extended sampler params after ${status}`);
          try {
            payload = buildPayload(req.body, candidate, { includeExtended: false });
            nimResponse = await callNim(payload, wantStream);
            servedBy = candidate;
            break;
          } catch (err2) {
            lastError = err2;
            if (!isModelUnavailable(err2.response?.status, await readErrorBody(err2))) throw err2;
            console.warn(`[proxy] ${candidate} unavailable, trying next model`);
            continue;
          }
        }

        lastError = err;
        if (isModelUnavailable(status, bodyText)) {
          console.warn(`[proxy] ${candidate} unavailable (${status}), trying next model`);
          continue; // model is down or not on this account: try the next one
        }
        throw err; // a real error (auth, rate limit, bad request): surface it
      }
    }

    if (!nimResponse) throw lastError || new Error('No NIM model available');
    if (servedBy !== nimModel) console.warn(`[proxy] served by fallback: ${servedBy} (wanted ${nimModel})`);

    // Always expose which model actually answered, so a silent fallback is visible.
    res.setHeader('X-Served-Model', servedBy);
    res.setHeader('X-Requested-Model', nimModel);

    // ---- streaming ----
    if (wantStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // stop proxies buffering the stream
      if (typeof res.flushHeaders === 'function') res.flushHeaders();

      let buffer = '';
      let openedThink = false;
      let inReasoning = false;

      // Degeneration tracking: how many times the same tiny delta has arrived in a row.
      let lastDelta = null;
      let repeatRun = 0;
      let aborted = false;

      nimResponse.data.on('data', (chunk) => {
        if (aborted) return;
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const raw of lines) {
          const line = raw.trim();
          if (!line.startsWith('data:')) continue;

          const body = line.slice(5).trim();
          if (body === '[DONE]') {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(body);
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              if (SHOW_REASONING) {
                const content = delta.content || '';
                const reasoning = delta.reasoning_content || '';
                let out = '';

                if (reasoning) {
                  if (!openedThink) { out += '<think>\n'; openedThink = true; }
                  inReasoning = true;
                  out += reasoning;
                }
                if (content) {
                  if (inReasoning) { out += '\n</think>\n\n'; inReasoning = false; }
                  out += content;
                }
                if (out) delta.content = out;
              }
              delete delta.reasoning_content; // non-standard; clients choke on it

              // Watch for the same tiny fragment arriving over and over.
              const piece = delta.content || '';
              if (piece && piece.trim() && piece.length <= 3) {
                if (piece === lastDelta) {
                  repeatRun++;
                } else {
                  lastDelta = piece;
                  repeatRun = 1;
                }
                if (repeatRun >= DEGEN_REPEATS) {
                  console.error(`[proxy] degenerate output detected (${JSON.stringify(piece)} x${repeatRun}) - aborting stream`);
                  aborted = true;
                  try { nimResponse.data.destroy(); } catch (_) {}
                  res.write(`data: ${JSON.stringify({
                    id: data.id,
                    object: 'chat.completion.chunk',
                    choices: [{ index: 0, delta: {}, finish_reason: 'length' }]
                  })}\n\n`);
                  res.write('data: [DONE]\n\n');
                  return res.end();
                }
              } else if (piece && piece.trim()) {
                lastDelta = null;
                repeatRun = 0;
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) {
            res.write(`${line}\n\n`); // pass through anything unparseable
          }
        }
      });

      nimResponse.data.on('end', () => {
        if (aborted || res.writableEnded) return; // degeneration guard already closed it
        if (SHOW_REASONING && inReasoning) {
          // Stream ended while still inside reasoning: close the tag so the client
          // does not render a dangling <think>.
          res.write(`data: ${JSON.stringify({
            choices: [{ index: 0, delta: { content: '\n</think>\n\n' }, finish_reason: null }]
          })}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });

      nimResponse.data.on('error', (err) => {
        if (aborted || res.writableEnded) return;
        console.error('[proxy] stream error:', err.message);
        res.end();
      });
      req.on('close', () => nimResponse.data.destroy?.());
      return;
    }

    // ---- non-streaming ----
    const choices = (nimResponse.data.choices || []).map((choice) => {
      let content = choice.message?.content || '';
      const reasoning = choice.message?.reasoning_content;

      // Strip a degenerate repeated tail so the client never renders a wall of junk.
      const degenAt = findDegenerateTail(content);
      if (degenAt > -1) {
        console.error(`[proxy] degenerate tail trimmed at char ${degenAt} of ${content.length}`);
        content = content.slice(0, degenAt).trimEnd();
      }

      if (SHOW_REASONING) {
        if (reasoning) {
          content = `<think>\n${reasoning}\n</think>\n\n${content}`;
        } else if (content.includes('</think>') && !content.trimStart().startsWith('<think>')) {
          content = `<think>\n${content}`;
        }
      }

      return {
        index: choice.index ?? 0,
        message: { role: choice.message?.role || 'assistant', content },
        finish_reason: choice.finish_reason ?? 'stop'
      };
    });

    res.json({
      id: nimResponse.data.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: servedBy,   // the model that actually answered, not the alias asked for
      choices,
      usage: nimResponse.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    });

  } catch (error) {
    const status = error.response?.status || 500;
    const upstream = error.response?.data;
    const message =
      upstream?.error?.message ||
      upstream?.detail ||
      (typeof upstream === 'string' ? upstream : null) ||
      error.message ||
      'Internal server error';

    console.error('[proxy] error', status, typeof upstream === 'object' ? JSON.stringify(upstream) : message);

    // If the stream already started we cannot send JSON; just close it.
    if (res.headersSent) return res.end();

    res.status(status).json({ error: { message, type: 'invalid_request_error', code: status } });
  }
});

// Final catch-all. Written as bare middleware rather than app.all('*') so it works on
// both Express 4 and Express 5 (v5's router rejects a bare '*' path).
app.use((req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

// Vercel imports the app; a normal host runs it directly.
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`OpenAI to NVIDIA NIM Proxy listening on ${PORT}`);
    console.log(`  health         : /health`);
    console.log(`  api key set    : ${Boolean(NIM_API_KEY)}`);
    console.log(`  reasoning shown: ${SHOW_REASONING}`);
    console.log(`  thinking mode  : ${ENABLE_THINKING_MODE}`);
  });
}

module.exports = app;
