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

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = true; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-flash',
  'gpt-4': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-opus': 'z-ai/glm-5.1',
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
        const { model, messages, temperature, max_tokens, stream } = req.body;

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
            temperature: temperature || 0.6,
            max_tokens: max_tokens || 9024,
            stream: stream || false,
            // Inject precise parameters for Qwen and GLM directly into the body root
            ...(ENABLE_THINKING_MODE ? {
                chat_template_kwargs: {
                    ...(nimModel.toLowerCase().includes('qwen') ? { thinking: true } : {}),
                    ...(nimModel.toLowerCase().includes('glm') ? { enable_thinking: true, clear_thinking: false } : {})
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
            let reasoningStarted = false;

            // FIX: Using nimResponse instead of 'response' to avoid 500 error
            nimResponse.data.on('data', (chunk) => {
                buffer += chunk.toString();
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        if (line.includes('[DONE]')) {
                            res.write(line + '\n');
                            return;
                        }
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.choices?.[0]?.delta) {
                                let content = data.choices[0].delta.content || '';
                                let reasoning = data.choices[0].delta.reasoning_content || '';
                            
                                if (SHOW_REASONING) {
                                    // 1. Capture the hidden reasoning_content field
                                    if (reasoning) {
                                        if (!reasoningStarted) {
                                            // First chunk of reasoning: inject the opening tag
                                            data.choices[0].delta.content = '<think>\n' + reasoning;
                                            reasoningStarted = true;
                                        } else {
                                            // Ongoing reasoning: pass it directly as visible content
                                            data.choices[0].delta.content = reasoning;
                                        }
                                    } 
                                    // 2. Capture when it transitions back to the main response content
                                    else if (content) {
                                        if (reasoningStarted) {
                                            // First chunk of actual response text: close the think tag safely
                                            data.choices[0].delta.content = '</think>\n\n' + content;
                                            reasoningStarted = false;
                                        } else {
                                            // Regular content stream continues normally
                                            data.choices[0].delta.content = content;
                                        }
                                    }
                                }
                            
                                // Safely remove the non-standard field so Janitor doesn't choke on it
                                delete data.choices[0].delta.reasoning_content;
                            }
                            res.write(`data: ${JSON.stringify(data)}\n\n`);
                        } catch (e) {
                            res.write(line + '\n');
                        }
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
                    if (SHOW_REASONING && choice.message?.reasoning_content) {
                        fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
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
