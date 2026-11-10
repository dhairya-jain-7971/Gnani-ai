"""LLM with tool calling. Always streams from the provider; the baseline mode simply does
not act on text until the stream ends, so instrumentation is identical in both modes.

Timestamps written into `ts` (seconds on the caller's monotonic clock):
  t_llm_request       first request sent (includes the tool-deciding round trip)
  t_tool_start/end    tool execution
  t_llm_final_request second request (after tool result), if a tool was called
  t_llm_first_token   first text token of the spoken answer
  t_llm_first_sentence is set by the pipeline when the chunker emits the first chunk
"""
import json

from . import config, tools

# v1 used 2 rounds: a question needing two sequential tool calls (scholarship + fees) ended with
# no spoken answer. Bench run 20261110-181229, clean_20, 6/6 runs.
MAX_ROUNDS = 4


def _provider():
    p = config.LLM_PROVIDER
    if p:
        return p
    if config.ANTHROPIC_API_KEY:
        return "anthropic"
    if config.OPENAI_API_KEY:
        return "openai"
    raise RuntimeError("No LLM key: set ANTHROPIC_API_KEY or OPENAI_API_KEY (see .env.example)")


class LLM:
    def __init__(self):
        self.provider = _provider()
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
            self.model = config.LLM_MODEL or "claude-haiku-5-5"
        else:
            import openai
            self.client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
            self.model = config.LLM_MODEL or "gpt-4o-mini"

    async def respond(self, user_text: str, on_delta, ts: dict, now) -> dict:
        """Returns {'text': str, 'tool_calls': [{'name','args','result'}], 'model': str}."""
        return await (self._anthropic if self.provider == "anthropic" else self._openai)(
            user_text, on_delta, ts, now)

    # ---------------- Anthropic ----------------
    async def _anthropic(self, user_text, on_delta, ts, now):
        specs = [{"name": s["name"], "description": s["description"], "input_schema": s["schema"]}
                 for s in tools.SPECS]
        messages = [{"role": "user", "content": user_text}]
        out, calls, first = [], [], True
        for round_no in range(MAX_ROUNDS):
            ts["t_llm_request" if round_no == 0 else "t_llm_final_request"] = now()
            async with self.client.messages.stream(
                model=self.model, max_tokens=config.LLM_MAX_TOKENS, temperature=config.LLM_TEMPERATURE,
                system=tools.SYSTEM_PROMPT, tools=specs, messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    if text and first:
                        ts["t_llm_first_token"], first = now(), False
                    if text:
                        out.append(text)
                        await on_delta(text)
                msg = await stream.get_final_message()
            uses = [b for b in msg.content if b.type == "tool_use"]
            if not uses:
                break
            messages.append({"role": "assistant", "content": msg.content})
            results = []
            ts["t_tool_start"] = now()
            for u in uses:
                r = tools.FUNCS[u.name](**u.input)
                calls.append({"name": u.name, "args": u.input, "result": r})
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(r)})
            ts["t_tool_end"] = now()
            messages.append({"role": "user", "content": results})
        return {"text": "".join(out), "tool_calls": calls, "model": self.model}

    # ---------------- OpenAI-compatible ----------------
    async def _openai(self, user_text, on_delta, ts, now):
        specs = [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                                  "parameters": s["schema"]}} for s in tools.SPECS]
        messages = [{"role": "system", "content": tools.SYSTEM_PROMPT},
                    {"role": "user", "content": user_text}]
        out, calls, first = [], [], True
        for round_no in range(MAX_ROUNDS):
            ts["t_llm_request" if round_no == 0 else "t_llm_final_request"] = now()
            extra = {"reasoning_effort": config.LLM_REASONING_EFFORT} if "gpt-oss" in self.model else {}
            stream = await self.client.chat.completions.create(
                model=self.model, messages=messages, tools=specs, stream=True,
                temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS, **extra)
            tcs = {}
            async for ev in stream:
                if not ev.choices:
                    continue
                d = ev.choices[0].delta
                if d.content:
                    if first:
                        ts["t_llm_first_token"], first = now(), False
                    out.append(d.content)
                    await on_delta(d.content)
                for tc in d.tool_calls or []:
                    slot = tcs.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    slot["id"] = tc.id or slot["id"]
                    slot["name"] = (tc.function.name or slot["name"]) if tc.function else slot["name"]
                    slot["args"] += (tc.function.arguments or "") if tc.function else ""
            if not tcs:
                break
            messages.append({"role": "assistant", "content": "".join(out) or None, "tool_calls": [
                {"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["args"]}}
                for t in tcs.values()]})
            ts["t_tool_start"] = now()
            for t in tcs.values():
                args = json.loads(t["args"] or "{}")
                r = tools.FUNCS[t["name"]](**args)
                calls.append({"name": t["name"], "args": args, "result": r})
                messages.append({"role": "tool", "tool_call_id": t["id"], "content": json.dumps(r)})
            ts["t_tool_end"] = now()
            out.clear()  # text emitted alongside a tool call is not the final answer
        return {"text": "".join(out), "tool_calls": calls, "model": self.model}
