(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.DPRCodingPlanAdapter = api;
  }
})(typeof self !== 'undefined' ? self : globalThis, function () {
  const DEFAULT_TIMEOUT_MS = 120000;
  const DEFAULT_SYSTEM_PROMPT =
    'You are a retrieval planning assistant and can only return valid JSON. '
    + 'The response must be fully based on the current user input and must not reference prior conversation history.';

  const normalizeText = (value) => String(value || '').trim();

  const normalizeProviderName = (value) => {
    const key = normalizeText(value).toLowerCase();
    if (!key) return '';
    if (['openai', 'gpt', 'chatgpt'].includes(key)) return 'openai';
    if (['deepseek'].includes(key)) return 'deepseek';
    if (['anthropic', 'claude'].includes(key)) return 'anthropic';
    if (['gemini', 'google', 'google-ai', 'googleai'].includes(key)) return 'gemini';
    if (['blt', 'bltcy', 'plato', 'gptbest'].includes(key)) return 'blt';
    if (['siliconflow', 'silicon-flow', 'sflow'].includes(key)) return 'siliconflow';
    if (['cstcloud', 'cst', 'cst-cloud', 'keji', 'keji-yun'].includes(key)) return 'cstcloud';
    if (['openrouter'].includes(key)) return 'openrouter';
    if (['ollama', 'local'].includes(key)) return 'ollama';
    if (['openai-compatible', 'openai_compatible', 'generic', 'compat'].includes(key)) {
      return 'openai-compatible';
    }
    return key;
  };

  const providerFamily = (provider) => {
    const normalized = normalizeProviderName(provider);
    if (normalized === 'anthropic') return 'anthropic';
    if (normalized === 'gemini') return 'gemini';
    return 'openai-compatible';
  };

  const inferProviderFromConfig = (config) => {
    const explicit = normalizeProviderName(config && config.provider);
    if (explicit) return explicit;

    const baseUrl = normalizeText(config && config.baseUrl).toLowerCase();
    const model = normalizeText(config && config.model).toLowerCase();

    if (baseUrl.includes('anthropic') || model.startsWith('claude')) return 'anthropic';
    if (baseUrl.includes('generativelanguage') || baseUrl.includes('googleapis.com') || model.startsWith('gemini')) {
      return 'gemini';
    }
    if (baseUrl.includes('openrouter')) return 'openrouter';
    if (baseUrl.includes('deepseek') || model.startsWith('deepseek')) return 'deepseek';
    if (baseUrl.includes('api.openai.com') || model.startsWith('gpt-') || model.startsWith('o1') || model.startsWith('o3') || model.startsWith('o4')) {
      return 'openai';
    }
    if (baseUrl.includes('bltcy') || baseUrl.includes('gptbest') || model.includes('gpt-5-chat')) return 'blt';
    if (baseUrl.includes('siliconflow')) return 'siliconflow';
    if (baseUrl.includes('cstcloud')) return 'cstcloud';
    if (baseUrl.includes('localhost') || baseUrl.includes('ollama')) return 'ollama';
    return 'openai-compatible';
  };

  const resolveDefaultBaseUrl = (provider) => {
    const normalized = normalizeProviderName(provider);
    if (normalized === 'openai') return 'https://api.openai.com/v1';
    if (normalized === 'deepseek') return 'https://api.deepseek.com/v1';
    if (normalized === 'anthropic') return 'https://api.anthropic.com/v1';
    if (normalized === 'gemini') return 'https://generativelanguage.googleapis.com/v1beta';
    return '';
  };

  const normalizeProviderConfig = (config) => {
    const provider = inferProviderFromConfig(config || {});
    const baseUrl = normalizeText((config && config.baseUrl) || resolveDefaultBaseUrl(provider)).replace(/\/+$/, '');
    const model = normalizeText(config && config.model);
    const apiKey = normalizeText(config && config.apiKey);
    const label = normalizeText(config && config.label) || model || provider;
    return {
      provider,
      family: providerFamily(provider),
      baseUrl,
      model,
      apiKey,
      label,
      enabled: config ? config.enabled !== false : true,
    };
  };

  const flattenLegacyProviderConfigs = (secret) => {
    const out = [];
    const summarized = secret && secret.summarizedLLM && typeof secret.summarizedLLM === 'object'
      ? secret.summarizedLLM
      : {};
    if (normalizeText(summarized.apiKey) && normalizeText(summarized.model)) {
      out.push({
        provider: summarized.provider,
        baseUrl: summarized.baseUrl,
        apiKey: summarized.apiKey,
        model: summarized.model,
        label: summarized.label || summarized.model,
      });
    }

    const chatLLMs = Array.isArray(secret && secret.chatLLMs) ? secret.chatLLMs : [];
    chatLLMs.forEach((item) => {
      if (!item || typeof item !== 'object') return;
      const apiKey = normalizeText(item.apiKey);
      const baseUrl = normalizeText(item.baseUrl);
      const provider = normalizeText(item.provider);
      const models = Array.isArray(item.models) ? item.models : [];
      models.forEach((model) => {
        const name = normalizeText(model);
        if (!apiKey || !name) return;
        out.push({
          provider,
          baseUrl,
          apiKey,
          model: name,
          label: name,
        });
      });
    });
    return out;
  };

  const listProviderConfigsFromSecret = (secret) => {
    const explicit = Array.isArray(secret && secret.codingPlanProviders)
      ? secret.codingPlanProviders
      : [];
    const rawList = explicit.length > 0 ? explicit : flattenLegacyProviderConfigs(secret || {});
    const seen = new Set();
    const out = [];
    rawList.forEach((item) => {
      const normalized = normalizeProviderConfig(item || {});
      if (!normalized.enabled || !normalized.model || !normalized.apiKey) return;
      const key = [
        normalized.provider,
        normalized.baseUrl,
        normalized.model,
        normalized.apiKey,
      ].join('|');
      if (seen.has(key)) return;
      seen.add(key);
      out.push(normalized);
    });
    return out;
  };

  const buildOpenAICompatibleEndpoints = (baseUrl) => {
    const out = [];
    const pushUnique = (value) => {
      const next = normalizeText(value);
      if (!next || out.includes(next)) return;
      out.push(next);
    };
    const src = normalizeText(baseUrl).replace(/\/+$/, '');
    if (!src) return out;
    if (src.includes('/chat/completions')) {
      pushUnique(src);
      pushUnique(src.replace(/\/chat\/completions$/, '/v1/chat/completions'));
      return out;
    }
    if (/\/v\d+(?:beta)?$/i.test(src)) {
      pushUnique(`${src}/chat/completions`);
      return out;
    }
    pushUnique(`${src}/v1/chat/completions`);
    pushUnique(`${src}/chat/completions`);
    return out;
  };

  const buildAnthropicEndpoint = (baseUrl) => {
    const src = normalizeText(baseUrl).replace(/\/+$/, '');
    if (!src) return 'https://api.anthropic.com/v1/messages';
    if (src.endsWith('/messages')) return src;
    if (/\/v\d+$/.test(src)) return `${src}/messages`;
    return `${src}/v1/messages`;
  };

  const appendQueryParam = (url, key, value) => {
    const joiner = url.includes('?') ? '&' : '?';
    return `${url}${joiner}${encodeURIComponent(key)}=${encodeURIComponent(value)}`;
  };

  const buildGeminiEndpoint = (baseUrl, model, apiKey) => {
    const src = normalizeText(baseUrl).replace(/\/+$/, '') || 'https://generativelanguage.googleapis.com/v1beta';
    if (src.includes(':generateContent')) {
      return src.includes('key=') ? src : appendQueryParam(src, 'key', apiKey);
    }
    if (src.includes('/models/')) {
      const url = `${src}/${encodeURIComponent(model)}:generateContent`;
      return appendQueryParam(url, 'key', apiKey);
    }
    const url = `${src}/models/${encodeURIComponent(model)}:generateContent`;
    return appendQueryParam(url, 'key', apiKey);
  };

  const buildProviderRequestDescriptors = (providerConfig, options = {}) => {
    const config = normalizeProviderConfig(providerConfig || {});
    if (!config.apiKey) {
      throw new Error('coding plan provider 缺少 apiKey。');
    }
    if (!config.model) {
      throw new Error('coding plan provider 缺少 model。');
    }
    const prompt = normalizeText(options.prompt);
    if (!prompt) {
      throw new Error('coding plan prompt 不能为空。');
    }
    const systemPrompt = normalizeText(options.systemPrompt) || DEFAULT_SYSTEM_PROMPT;
    const temperature = typeof options.temperature === 'number' ? options.temperature : 0.1;

    if (config.family === 'anthropic') {
      const url = buildAnthropicEndpoint(config.baseUrl);
      return [
        {
          provider: config.provider,
          model: config.model,
          family: config.family,
          url,
          init: {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'x-api-key': config.apiKey,
              'anthropic-version': '2023-06-01',
            },
            body: JSON.stringify({
              model: config.model,
              max_tokens: 4096,
              temperature,
              system: systemPrompt,
              messages: [
                {
                  role: 'user',
                  content: prompt,
                },
              ],
            }),
          },
        },
      ];
    }

    if (config.family === 'gemini') {
      const url = buildGeminiEndpoint(config.baseUrl, config.model, config.apiKey);
      return [
        {
          provider: config.provider,
          model: config.model,
          family: config.family,
          url,
          init: {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              systemInstruction: {
                parts: [{ text: systemPrompt }],
              },
              contents: [
                {
                  role: 'user',
                  parts: [{ text: prompt }],
                },
              ],
              generationConfig: {
                temperature,
                responseMimeType: 'application/json',
              },
            }),
          },
        },
      ];
    }

    const endpoints = buildOpenAICompatibleEndpoints(config.baseUrl || resolveDefaultBaseUrl(config.provider));
    if (!endpoints.length) {
      throw new Error('coding plan provider 缺少 baseUrl。');
    }
    const headers = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      Authorization: `Bearer ${config.apiKey}`,
      'x-api-key': config.apiKey,
    };
    const basePayload = {
      model: config.model,
      messages: [
        {
          role: 'system',
          content: systemPrompt,
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature,
    };
    const descriptors = [];
    endpoints.forEach((url) => {
      descriptors.push({
        provider: config.provider,
        model: config.model,
        family: config.family,
        url,
        init: {
          method: 'POST',
          headers,
          body: JSON.stringify({
            ...basePayload,
            response_format: { type: 'json_object' },
          }),
        },
      });
      descriptors.push({
        provider: config.provider,
        model: config.model,
        family: config.family,
        url,
        init: {
          method: 'POST',
          headers,
          body: JSON.stringify(basePayload),
        },
      });
    });
    return descriptors;
  };

  const normalizeOpenAIContent = (content) => {
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      return content
        .map((part) => {
          if (typeof part === 'string') return part;
          if (!part || typeof part !== 'object') return '';
          return normalizeText(part.text || part.content || part.output_text || '');
        })
        .filter(Boolean)
        .join('\n');
    }
    if (content && typeof content === 'object') {
      return normalizeText(content.text || content.content || content.output_text || '');
    }
    return '';
  };

  const extractContentFromResponse = (provider, data) => {
    const family = providerFamily(provider);
    if (family === 'anthropic') {
      const content = Array.isArray(data && data.content) ? data.content : [];
      return content
        .map((item) => normalizeText(item && item.text))
        .filter(Boolean)
        .join('\n');
    }
    if (family === 'gemini') {
      const candidates = Array.isArray(data && data.candidates) ? data.candidates : [];
      const first = candidates[0] || {};
      const parts = Array.isArray(first && first.content && first.content.parts)
        ? first.content.parts
        : [];
      return parts
        .map((part) => normalizeText(part && part.text))
        .filter(Boolean)
        .join('\n');
    }
    const firstChoice = (((data || {}).choices || [])[0] || {});
    const message = firstChoice.message || {};
    const content = normalizeOpenAIContent(message.content);
    if (content) return content;
    return normalizeText((data && data.output_text) || '');
  };

  const requestCodingPlan = async (options = {}) => {
    const providerConfig = normalizeProviderConfig(options.providerConfig || {});
    const descriptors = buildProviderRequestDescriptors(providerConfig, options);
    const fetchImpl = typeof options.fetchImpl === 'function' ? options.fetchImpl : fetch.bind(globalThis);
    const timeoutMs = Number.isFinite(options.timeoutMs) ? Number(options.timeoutMs) : DEFAULT_TIMEOUT_MS;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    let lastError = null;
    try {
      for (let i = 0; i < descriptors.length; i += 1) {
        const descriptor = descriptors[i];
        try {
          const res = await fetchImpl(descriptor.url, {
            ...descriptor.init,
            signal: controller.signal,
          });
          const text = await res.text().catch(() => '');
          if (!res.ok) {
            lastError = new Error(`HTTP ${res.status} ${text || res.statusText}`);
            continue;
          }
          const data = text ? JSON.parse(text) : {};
          const content = extractContentFromResponse(providerConfig.provider, data);
          if (!content) {
            lastError = new Error('coding plan 响应缺少可解析内容。');
            continue;
          }
          return {
            provider: providerConfig.provider,
            model: providerConfig.model,
            family: providerConfig.family,
            content,
            raw: data,
          };
        } catch (error) {
          if (error && error.name === 'AbortError') {
            throw new Error('coding plan 请求超时，请稍后重试。');
          }
          lastError = error;
        }
      }
    } finally {
      clearTimeout(timer);
    }

    if (lastError) {
      throw lastError;
    }
    throw new Error('coding plan 请求失败。');
  };

  return {
    DEFAULT_SYSTEM_PROMPT,
    inferProviderFromConfig,
    normalizeProviderConfig,
    listProviderConfigsFromSecret,
    buildProviderRequestDescriptors,
    extractContentFromResponse,
    requestCodingPlan,
  };
});
