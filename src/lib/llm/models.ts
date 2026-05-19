import type { ModelOption } from '../../types'

/**
 * Curated default models. Users can also enter a custom model id.
 * Models default to OpenRouter ids; the openai provider should select an
 * "openai/*" or a bare OpenAI id.
 */
export const DEFAULT_MODELS: ModelOption[] = [
  {
    id: 'openrouter/auto',
    name: 'OpenRouter Auto',
    description: 'Auto-routes to the best model for the prompt',
    iconKey: 'bolt',
    badge: 'Default',
  },
  {
    id: 'openrouter/free',
    name: 'OpenRouter Free',
    description: 'Routes to a free-tier model automatically',
    iconKey: 'zap',
    badge: 'Fast',
  },
  {
    id: 'openai/gpt-oss-120b:free',
    name: 'GPT-OSS 120B',
    description: 'OpenAI open-weights, free tier',
    iconKey: 'brain',
  },
  {
    id: 'nvidia/nemotron-3-super-120b-a12b:free',
    name: 'Nemotron 3 Super 120B',
    description: 'NVIDIA MoE, free tier',
    iconKey: 'zap',
    badge: 'Fast',
  },
  {
    id: 'google/gemma-4-26b-a4b-it:free',
    name: 'Gemma 4 26B',
    description: 'Google instruct, free tier',
    iconKey: 'sparkles',
  },
  {
    id: 'nousresearch/hermes-3-llama-3.1-405b:free',
    name: 'Hermes 3 405B',
    description: 'Nous Research, free tier',
    iconKey: 'sparkles',
    badge: 'Pro',
  },
]
