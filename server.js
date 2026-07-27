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

// Applied ONLY when the client omits the value (or sends 0, see ZERO_MEANS_UNSET).
const DEFAULTS = {
  temperature:       0.8,   // GLM 5.2 community sweet spot
  top_p:             0.95,  // GLM default; avoid tuning this and temperature together
  max_tokens:        1024,  // keeps replies tight and avoids Cloudflare 524 timeouts
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

// Samplers SillyTavern offers that NIM does not implement. Dropped silently so they
// never trigger a 400. (ST's docs note these have no effect on stricter endpoints.)
const UNSUPPORTED = [
  'top_a', 'typical_p', 'tfs', 'epsilon_cutoff', 'eta_cutoff', 'mirostat',
  'mirostat_tau', 'mirostat_eta', 'smoothing_factor', 'smoothing_curve',
  'dynatemp', 'dynatemp_low', 'dynatemp_high', 'encoder_repetition_penalty',
  'no_repeat_ngram_size', 'penalty_alpha', 'guidance_scale', 'repetition_penalty_range'
];

const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'deepseek-ai/deepseek-v4-flash',
  'gpt-4':          'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo':    'moonshotai/kimi-k2.6',
  'gpt-4o':         'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-opus':  'z-ai/glm-5.2',
  'claude-3-sonnet':'deepseek-ai/deepseek-v4-pro',
  'gemini-pro':     'mistralai/mistral-large-3-675b-instruct-2512'
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

  // max_tokens of 0 or less means "unlimited" in some clients; clamp it so replies
  // cannot run forever and trip a gateway timeout.
  if (!payload.max_tokens || payload.max_tokens <= 0) payload.max_tokens = DEFAULTS.max_tokens;

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

  return payload;
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

    let nimResponse;
    try {
      nimResponse = await callNim(payload, wantStream);
    } catch (err) {
      // NIM rejected something. Most often that is an extended sampler this model
      // does not implement, so drop them and try once more before giving up.
      const status = err.response?.status;
      const hadExtended = EXTENDED_PARAMS.some((k) => payload[k] !== undefined);
      if ((status === 400 || status === 422) && hadExtended) {
        console.warn('[proxy] retrying without extended sampler params after', status);
        payload = buildPayload(req.body, nimModel, { includeExtended: false });
        nimResponse = await callNim(payload, wantStream);
      } else {
        throw err;
      }
    }

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

      nimResponse.data.on('data', (chunk) => {
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
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) {
            res.write(`${line}\n\n`); // pass through anything unparseable
          }
        }
      });

      nimResponse.data.on('end', () => {
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
      model: req.body.model,
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
