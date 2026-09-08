// Standalone transport for the frozen judge; no dependency on a private project.
// Protocol: https://ai.google.dev/gemini-api/docs/openai
let input = '';
for await (const chunk of process.stdin) input += chunk;
const request = JSON.parse(input);
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) throw new Error('Set GEMINI_API_KEY before running paid judging.');
const model = 'gemini-3.5-flash';
const baseUrl = process.env.GEMINI_BASE_URL || 'https://generativelanguage.googleapis.com/v1beta/openai';
const body = {
  model,
  messages: [
    { role: 'system', content: request.rubric },
    { role: 'user', content: JSON.stringify(request.payload) },
  ],
  response_format: { type: 'json_schema', json_schema: { name: 'medical_answer_judgment', strict: false, schema: request.schema } },
};
if (request.thinking_level !== undefined) body.reasoning_effort = request.thinking_level;
if (request.max_output_tokens !== undefined) body.max_tokens = request.max_output_tokens;
const started = performance.now();
const response = await fetch(`${baseUrl.replace(/\/$/, '')}/chat/completions`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
  body: JSON.stringify(body),
  signal: AbortSignal.timeout(80000),
});
if (!response.ok) throw new Error(`Gemini HTTP ${response.status}: ${(await response.text()).slice(0, 2000)}`);
const metadata = await response.json();
if (!metadata.choices?.length) throw new Error('Gemini returned no choices.');
process.stdout.write(JSON.stringify({
  requested_model: model,
  returned_model: metadata.model,
  usage: metadata.usage,
  finish_reason: metadata.choices[0].finish_reason,
  elapsed_seconds: (performance.now() - started) / 1000,
  thinking_level: request.thinking_level,
  message: metadata.choices[0].message,
}));
