const assert = require('node:assert/strict');

const adapter = require('../app/coding-plan-adapter.js');

assert.equal(
  adapter.inferProviderFromConfig({ model: 'claude-3-7-sonnet-latest' }),
  'anthropic',
);
assert.equal(
  adapter.inferProviderFromConfig({ model: 'gemini-2.5-flash' }),
  'gemini',
);
assert.equal(
  adapter.inferProviderFromConfig({ baseUrl: 'https://api.deepseek.com/v1' }),
  'deepseek',
);

const configs = adapter.listProviderConfigsFromSecret({
  chatLLMs: [
    {
      apiKey: 'k',
      baseUrl: 'https://api.openai.com/v1',
      models: ['gpt-5-mini', 'gpt-5-mini'],
    },
  ],
});
assert.equal(configs.length, 1);
assert.equal(configs[0].provider, 'openai');

const descriptors = adapter.buildProviderRequestDescriptors(
  {
    provider: 'gemini',
    model: 'gemini-2.5-flash',
    apiKey: 'k',
  },
  {
    prompt: 'hello',
  },
);
assert.equal(descriptors.length, 1);
assert.ok(descriptors[0].url.includes(':generateContent?key=k'));

const text = adapter.extractContentFromResponse('anthropic', {
  content: [{ text: '{"tag":"SR"}' }],
});
assert.equal(text, '{"tag":"SR"}');

console.log('coding plan adapter tests passed');
