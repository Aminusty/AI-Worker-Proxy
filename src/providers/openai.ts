import OpenAI from 'openai';
import { BaseProvider } from './base';
import { OpenAIChatRequest, ProviderResponse } from '../types';

export class OpenAIProvider extends BaseProvider {
  constructor(
    model: string,
    baseUrl?: string,
    private extraBody?: Record<string, unknown>
  ) {
    super(model, baseUrl);
  }

  async chat(request: OpenAIChatRequest, apiKey: string): Promise<ProviderResponse> {
    try {
      const client = new OpenAI({
        apiKey,
        baseURL: this.baseUrl,
      });

      const params:
        | OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
        | OpenAI.Chat.ChatCompletionCreateParamsStreaming = {
        ...(this.extraBody ?? {}),
        model: this.model,
        messages: request.messages as any,
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        top_p: request.top_p,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        stop: request.stop,
        stream: request.stream || false,
        tools: request.tools as any,
        tool_choice: request.tool_choice as any,
      };

      if (request.stream) {
        return this.handleStream(client, params as OpenAI.Chat.ChatCompletionCreateParamsStreaming);
      }
      return this.handleNonStream(
        client,
        params as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
      );
    } catch (error) {
      return this.handleError(error, 'OpenAIProvider');
    }
  }

  private async handleNonStream(
    client: OpenAI,
    params: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
  ): Promise<ProviderResponse> {
    const response = await client.chat.completions.create(params);
    return { success: true, response: response as any };
  }

  private async handleStream(
    client: OpenAI,
    params: OpenAI.Chat.ChatCompletionCreateParamsStreaming
  ): Promise<ProviderResponse> {
    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();
    const startedAt = Date.now();
    let firstChunkLogged = false;
    let firstHeartbeatLogged = false;

    console.log('[OpenAIProvider] Starting NVIDIA stream request:', {
      model: params.model,
      timestamp: new Date(startedAt).toISOString(),
    });

    // Return the response stream immediately instead of waiting for NVIDIA's
    // first token. This lets Cloudflare send HTTP headers before a slow model
    // finishes its initial reasoning.
    void (async () => {
      const heartbeat = setInterval(() => {
        if (!firstHeartbeatLogged) {
          firstHeartbeatLogged = true;
          console.log('[OpenAIProvider] First heartbeat sent:', {
            elapsed_ms: Date.now() - startedAt,
          });
        }
        void writer.write(encoder.encode(': keep-alive\n\n')).catch(() => {
          // The client may have disconnected.
        });
      }, 15000);

      let phase: 'opening' | 'reading' = 'opening';

      try {
        console.log('[OpenAIProvider] Awaiting NVIDIA stream response:', {
          elapsed_ms: Date.now() - startedAt,
        });
        const stream = await client.chat.completions.create(params);
        console.log('[OpenAIProvider] NVIDIA stream opened:', {
          elapsed_ms: Date.now() - startedAt,
        });
        phase = 'reading';

        for await (const chunk of stream) {
          if (!firstChunkLogged) {
            firstChunkLogged = true;
            console.log('[OpenAIProvider] First NVIDIA chunk received:', {
              elapsed_ms: Date.now() - startedAt,
            });
          }
          await writer.write(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
        }
        await writer.write(encoder.encode('data: [DONE]\n\n'));
      } catch (error) {
        const apiError = error as Record<string, unknown>;
        const cause = apiError?.cause as Record<string, unknown> | undefined;
        const details = {
          phase,
          elapsed_ms: Date.now() - startedAt,
          name: typeof apiError?.name === 'string' ? apiError.name : null,
          constructor:
            error && typeof error === 'object'
              ? error.constructor?.name ?? null
              : null,
          status: apiError?.status ?? null,
          code: apiError?.code ?? null,
          type: apiError?.type ?? null,
          request_id: apiError?.request_id ?? null,
          message:
            typeof apiError?.message === 'string'
              ? apiError.message
              : String(error),
          cause_name: typeof cause?.name === 'string' ? cause.name : null,
          cause_message:
            typeof cause?.message === 'string' ? cause.message : null,
          own_properties:
            error && typeof error === 'object'
              ? Object.getOwnPropertyNames(error)
              : [],
        };

        // Use info-level logging as well so the diagnostic is visible even if
        // the observability view filters error-level console entries.
        console.log('[OpenAIProvider] NVIDIA request failed:', details);
        console.error('[OpenAIProvider] Stream error:', details);

        try {
          const errorFinish = {
            id: 'chatcmpl-error',
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: params.model,
            choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
          };
          await writer.write(encoder.encode(`data: ${JSON.stringify(errorFinish)}\n\n`));
          await writer.write(encoder.encode('data: [DONE]\n\n'));
        } catch {
          // Writer may already be closed.
        }
      } finally {
        clearInterval(heartbeat);
        try {
          await writer.close();
        } catch {
          // Already closed.
        }
      }
    })();

    return { success: true, stream: readable };
  }
}
