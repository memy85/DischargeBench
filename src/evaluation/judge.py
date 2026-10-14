"""
judge.py — the LLM judge client shared by every evaluation stage.

make_judge_client() returns an OpenAI-compatible client: the local vLLM server on
eval_port when one is set, otherwise the OpenAI API. structured_call() runs one
judge call with guided decoding into a Pydantic schema.
"""

import asyncio
import json

import openai
from openai import AsyncOpenAI
from pydantic import ValidationError

# A long client timeout so a slow-but-not-hung request is not aborted; a default
# output cap so a runaway generation cannot hang the server for minutes.
_JUDGE_TIMEOUT = 1800.0            # seconds
_DEFAULT_MAX_OUT = 8192            # CLSA passes 16384 explicitly
_TRANSIENT = (openai.APITimeoutError, openai.APIConnectionError,
              openai.InternalServerError, openai.RateLimitError)


def make_judge_client(eval_port: int | None = None) -> AsyncOpenAI:
    if eval_port:
        return AsyncOpenAI(base_url=f"http://localhost:{eval_port}/v1", api_key="EMPTY",
                           timeout=_JUDGE_TIMEOUT)
    return AsyncOpenAI(timeout=_JUDGE_TIMEOUT)


async def structured_call(client, model, prompt, schema, *,
                          temperature=0, max_output_tokens=None):
    """Run one structured judge call (client.responses.parse into `schema`).

    Returns None after 3 schema-invalid attempts; callers abstain on that unit.
    A degenerate vLLM response (output=None, surfacing as AttributeError) is retried
    once, then abstains. Transient API errors (timeout, connection, 5xx, rate limit)
    are retried with backoff, then abstain. Auth and other errors raise.
    """
    for attempt in range(5):
        try:
            out = await client.responses.parse(
                model=model,
                input=[{"role": "user", "content": prompt}],
                text_format=schema,
                temperature=temperature,
                max_output_tokens=(max_output_tokens or _DEFAULT_MAX_OUT),
            )
            return out.output_parsed
        except (ValidationError, json.JSONDecodeError):
            if attempt >= 2:
                return None
        except AttributeError as e:
            if attempt >= 1:
                print(f"  [ABSTAIN] {model} null/degenerate structured response "
                      f"({type(e).__name__}) — skipping this unit.")
                return None
        except _TRANSIENT as e:
            if attempt >= 4:
                print(f"  [ABSTAIN] {model} transient API error ({type(e).__name__}) "
                      f"after retries — skipping this unit.")
                return None
            await asyncio.sleep(min(2 ** attempt, 30))
    return None
