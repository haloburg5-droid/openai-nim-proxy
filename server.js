// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({
  limit: '20mb'
}));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // hide <think> reasoning from the chat output

// THINKING MODE TOGGLE - Enables thinking for models that support it
// OFF by default: forcing thinking on GLM with a heavy prompt (e.g. KERNEL) is
// what makes it loop inside <think>. Set back to true only if you want it.
const ENABLE_THINKING_MODE = false;

// Use the value the client (JanitorAI) sends; if it is missing or 0, fall back
// to an anti-repetition default so replies do not lock into the same phrasing.
const pick = (v, def) => (v === undefined || v === null || v === 0) ? def : v;

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-flash',
  'gpt-4': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-opus': 'z-ai/glm-5.2',
  'claude-3-sonnet': 'deepseek-ai/deepseek-v4-pro',
  'gemini-pro': 'mistralai/mistral-large-3-675b-instruct-2512'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
    try {
        // Pull the sampling params too, so JanitorAI's sliders actually reach NIM.
        const {
            model, messages, temperature, max_tokens, stream,
            top_p, top_k, frequency_penalty, presence_penalty, repetition_penalty
        } = req.body;

        // Smart model selection with fallback
        let nimModel = MODEL_MAPPING[model];
        if (!nimModel) {
            try {
                await axios.post(`${NIM_API_BASE}/chat/completions`, {
                    model: model,
                    messages: [{ role: 'user', content: 'test' }],
                    max_tokens: 1
                }, {
                    headers: {
                        'Authorization': `Bearer ${NIM_API_KEY}`,
                        'Content-Type': 'application/json'
                    },
                    validateStatus: (status) => status < 500
                }).then(apiRes => {
                    if (apiRes.status >= 200 && apiRes.status < 300) {
                        nimModel = model;
                    }
                });
            } catch (e) {}

            if (!nimModel) {
                const modelLower = model.toLowerCase();
                if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
                    nimModel = 'meta/llama-3.1-405b-instruct';
                } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
                    nimModel = 'meta/llama-3.1-70b-instruct';
                } else {
                    nimModel = 'meta/llama-3.1-8b-instruct';
                }
            }
        }

        // Build the raw HTTP request payload for NVIDIA NIM
        const nimRequest = {
          model: nimModel,
          messages: messages,
          temperature: temperature || 0.9,
          max_tokens: (max_tokens && max_tokens > 0) ? max_tokens : 2048,
          stream: stream || false,

          // --- Anti-repetition sampling (this is the part that was missing) ---
          top_p: pick(top_p, 0.9),
          frequency_penalty: pick(frequency_penalty, 0.5), // main lever: punishes recurring tokens like "he said it"
          presence_penalty: pick(presence_penalty, 0.4),   // nudges toward new vocabulary
          // top_k: pick(top_k, 60),                        // uncomment if your NIM model accepts top_k
          // repetition_penalty: pick(repetition_penalty, 1.1), // uncomment if accepted (non-OpenAI; some NIM models 422 on this)

          // Inject precise parameters for Qwen and GLM directly into chat_template_kwargs
          ...(ENABLE_THINKING_MODE ? {
            chat_template_kwargs: {
              ...(nimModel.toLowerCase().includes('qwen') ? { thinking: true } : {}),
              ...(nimModel.toLowerCase().includes('glm') ? { thinking: true } : {})
            }
          } : {})
        };

        // Fire the HTTP request to NVIDIA NIM
        const nimResponse = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
            headers: {
                'Authorization': `Bearer ${NIM_API_KEY}`,
                'Content-Type': 'application/json'
            },
            responseType: stream ? 'stream' : 'json'
        });

        if (stream) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');

            let buffer = '';
            let started = false;
            let inReasoningField = false;

            nimResponse.data.on('data', (chunk) => {
                buffer += chunk.toString();
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                lines.forEach(line => {
                    if (!line.startsWith('data: ')) return;

                    if (line.includes('[DONE]')) {
                        res.write(line + '\n');
                        return;
                    }

                    try {
                        const data = JSON.parse(line.slice(6));
                        const delta = data.choices?.[0]?.delta;

                        if (delta && SHOW_REASONING) {
                            const content = delta.content || '';
                            const reasoning = delta.reasoning_content || '';
                            let out = '';

                            if (reasoning) {
                                if (!started) {
                                    out += '<think>\n';
                                    started = true;
                                }
                                inReasoningField = true;
                                out += reasoning;
                            }

                            if (content) {
                                if (!started) {
                                    if (!content.trimStart().startsWith('<think>')) {
                                        out += '<think>\n';
                                    }
                                    started = true;
                                } else if (inReasoningField) {
                                    out += '</think>\n\n';
                                    inReasoningField = false;
                                }
                                out += content;
                            }

                            delta.content = out;
                        }

                        // Remove the non-standard property before sending to the client.
                        if (delta) delete delta.reasoning_content;

                        res.write(`data: ${JSON.stringify(data)}\n\n`);
                    } catch (e) {
                        res.write(line + '\n');
                    }
                });
            });

            nimResponse.data.on('end', () => res.end());
            nimResponse.data.on('error', (err) => {
                console.error('Stream error:', err);
                res.end();
            });

        } else {
            // Non-streaming response block handling
            const openaiResponse = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion',
                created: Math.floor(Date.now() / 1000),
                model: model,
                choices: nimResponse.data.choices.map(choice => {
                    let fullContent = choice.message?.content || '';
                    const reasoning = choice.message?.reasoning_content;

                    if (SHOW_REASONING) {
                        if (reasoning) {
                            fullContent = '<think>\n' + reasoning + '\n</think>\n\n' + fullContent;
                        } else if (fullContent.includes('</think>') && !fullContent.trimStart().startsWith('<think>')) {
                            fullContent = '<think>\n' + fullContent;
                        }
                    }

                    return {
                        index: choice.index,
                        message: {
                            role: choice.message.role,
                            content: fullContent
                        },
                        finish_reason: choice.finish_reason
                    };
                }),
                usage: nimResponse.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
            };
            res.json(openaiResponse);
        }

    } catch (error) {
        if (error.response && error.response.data) {
            console.error('NVIDIA API Error Response:', JSON.stringify(error.response.data));
        } else {
            console.error('Proxy error:', error.message);
        }
        res.status(error.response?.status || 500).json({
            error: {
                message: error.response?.data?.error?.message || error.message || 'Internal server error',
                type: 'invalid_request_error',
                code: error.response?.status || 500
            }
        });
    }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
