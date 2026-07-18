import time
import random

from llmpivot import LLMPivot
from llmpivot import PivotConfig

from gpe.helper.logger import Logger


class LLMWrapperError(RuntimeError):
    def __init__(self, operation, original_error):
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"LLM {operation} failed: {type(original_error).__name__}: {original_error}")


class LLMWrapper:
    def __init__(
        self,
        logger: Logger,
        model_id,
        base_url,
        api_key=None,
        model_type="online",
        stream=True,
        retries=1,
        retry_delay=1.0,
        retry_max_delay=60.0,
        retry_server_delay=3.0,
        **config_kwargs,
    ):
        self.logger = logger
        self.stream = stream
        self.retries = retries
        self.retry_delay = retry_delay
        self.retry_max_delay = retry_max_delay
        self.retry_server_delay = retry_server_delay
        self.config = PivotConfig(
            model_type=model_type,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            **config_kwargs,
        )
        self.llm = LLMPivot(self.config)
        self.usage = self._empty_usage()

    def _empty_usage(self):
        return {
            "request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "by_call_type": {},
        }

    def reset_usage(self):
        self.usage = self._empty_usage()

    def usage_snapshot(self):
        return {
            "request_count": self.usage["request_count"],
            "prompt_tokens": self.usage["prompt_tokens"],
            "completion_tokens": self.usage["completion_tokens"],
            "total_tokens": self.usage["total_tokens"],
            "by_call_type": {
                call_type: dict(values)
                for call_type, values in self.usage["by_call_type"].items()
            },
        }

    def usage_delta(self, before, after=None):
        after = after or self.usage_snapshot()
        delta = {
            "request_count": after["request_count"] - before["request_count"],
            "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
            "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
            "total_tokens": after["total_tokens"] - before["total_tokens"],
            "by_call_type": {},
        }
        call_types = set(before["by_call_type"]) | set(after["by_call_type"])
        for call_type in sorted(call_types):
            before_values = before["by_call_type"].get(call_type, self._empty_usage())
            after_values = after["by_call_type"].get(call_type, self._empty_usage())
            delta["by_call_type"][call_type] = {
                "request_count": after_values["request_count"] - before_values["request_count"],
                "prompt_tokens": after_values["prompt_tokens"] - before_values["prompt_tokens"],
                "completion_tokens": after_values["completion_tokens"] - before_values["completion_tokens"],
                "total_tokens": after_values["total_tokens"] - before_values["total_tokens"],
            }
        return delta

    def _record_usage(self, response, call_type):
        usage = response.get("usage") if isinstance(response, dict) else None
        if not usage:
            return
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        self.usage["request_count"] += 1
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += total_tokens
        by_type = self.usage["by_call_type"].setdefault(call_type, self._empty_usage())
        by_type["request_count"] += 1
        by_type["prompt_tokens"] += prompt_tokens
        by_type["completion_tokens"] += completion_tokens
        by_type["total_tokens"] += total_tokens

    def add_usage(self, usage_delta):
        if not usage_delta:
            return
        self.usage["request_count"] += int(usage_delta.get("request_count", 0) or 0)
        self.usage["prompt_tokens"] += int(usage_delta.get("prompt_tokens", 0) or 0)
        self.usage["completion_tokens"] += int(usage_delta.get("completion_tokens", 0) or 0)
        self.usage["total_tokens"] += int(usage_delta.get("total_tokens", 0) or 0)
        for call_type, values in dict(usage_delta.get("by_call_type", {})).items():
            by_type = self.usage["by_call_type"].setdefault(call_type, self._empty_usage())
            by_type["request_count"] += int(values.get("request_count", 0) or 0)
            by_type["prompt_tokens"] += int(values.get("prompt_tokens", 0) or 0)
            by_type["completion_tokens"] += int(values.get("completion_tokens", 0) or 0)
            by_type["total_tokens"] += int(values.get("total_tokens", 0) or 0)

    def clone_for_worker(self):
        config = self.config.model_dump() if hasattr(self.config, "model_dump") else dict(self.config.__dict__)
        return LLMWrapper(
            self.logger,
            model_id=config.pop("model_id"),
            base_url=config.pop("base_url"),
            api_key=config.pop("api_key", None),
            model_type=config.pop("model_type", "online"),
            stream=self.stream,
            retries=self.retries,
            retry_delay=self.retry_delay,
            retry_max_delay=self.retry_max_delay,
            retry_server_delay=self.retry_server_delay,
            **config,
        )

    def _is_server_or_network_error(self, error):
        text = str(error).lower()
        return (
            "error code: 500" in text
            or "'code': 500" in text
            or '"code": 500' in text
            or "network unstable" in text
            or "timeout" in text
            or "temporarily unavailable" in text
        )


    def _response_text(self, response):
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            for key in ("content", "text", "output_text"):
                value = response.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            message = response.get("message")
            if isinstance(message, dict):
                value = message.get("content")
                if isinstance(value, str) and value.strip():
                    return value
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    value = choice.get("text")
                    if isinstance(value, str) and value.strip():
                        return value
                    message = choice.get("message")
                    if isinstance(message, dict):
                        value = message.get("content")
                        if isinstance(value, str) and value.strip():
                            return value
            output = response.get("output")
            if isinstance(output, list):
                chunks = []
                for item in output:
                    if isinstance(item, dict):
                        for content_item in item.get("content", []) or []:
                            if isinstance(content_item, dict):
                                value = content_item.get("text") or content_item.get("content")
                                if isinstance(value, str):
                                    chunks.append(value)
                if chunks:
                    return "\n".join(chunks)
        for attr in ("content", "text", "output_text"):
            value = getattr(response, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _stream_response(self, messages, **kwargs):
        kwargs.pop("stream", None)
        chunks = self.llm.stream_generate(messages, **kwargs)
        content = "".join(str(chunk) for chunk in chunks if chunk is not None)
        usage = getattr(getattr(self.llm, "llm", None), "last_stream_usage", None)
        response = {"content": content}
        if usage:
            response["usage"] = usage
        return response

    def _retry_sleep_seconds(self, error, attempt):
        base_delay = float(self.retry_delay)
        if self._is_server_or_network_error(error):
            base_delay = max(base_delay, float(self.retry_server_delay))
        sleep_seconds = min(float(self.retry_max_delay), base_delay * (2 ** max(0, attempt - 1)))
        jitter = random.uniform(0.0, min(1.0, sleep_seconds * 0.1))
        return sleep_seconds + jitter

    def _sleep_before_retry(self, operation, error, attempt):
        sleep_seconds = self._retry_sleep_seconds(error, attempt)
        self.logger.warning(
            "LLM %s failed on attempt %s/%s, retrying in %.1fs",
            operation,
            attempt,
            self.retries,
            sleep_seconds,
        )
        self.logger.log_exception(error)
        time.sleep(sleep_seconds)

    def dialogue(self, messages, **kwargs):
        attempt = 0
        while True:
            try:
                if self.stream:
                    response = self._stream_response(messages, **kwargs)
                else:
                    kwargs["stream"] = False
                    response = self.llm.generate(messages, **kwargs)
                self._record_usage(response, "dialogue")
                return self._response_text(response)
            except Exception as error:
                if attempt >= self.retries:
                    raise LLMWrapperError("dialogue", error) from error
                attempt += 1
                self._sleep_before_retry("dialogue", error, attempt)

    def generate(self, messages, **kwargs):
        attempt = 0
        while True:
            try:
                if self.stream:
                    response = self._stream_response(messages, **kwargs)
                else:
                    kwargs["stream"] = False
                    response = self.llm.generate(messages, **kwargs)
                self._record_usage(response, "generate")
                return response
            except Exception as error:
                if attempt >= self.retries:
                    raise LLMWrapperError("generate", error) from error
                attempt += 1
                self._sleep_before_retry("generate", error, attempt)

    def stream_generate(self, messages, **kwargs):
        kwargs.setdefault("stream", self.stream)
        attempt = 0
        while True:
            try:
                return self.llm.stream_generate(messages, **kwargs)
            except Exception as error:
                if attempt >= self.retries:
                    raise LLMWrapperError("stream_generate", error) from error
                attempt += 1
                self._sleep_before_retry("stream_generate", error, attempt)

    def call_function(self, messages, tools, **kwargs):
        kwargs["stream"] = False
        kwargs.setdefault("tool_choice", "auto")
        attempt = 0
        while True:
            try:
                response = self.llm.generate(messages, tools=tools, **kwargs)
                self._record_usage(response, "call_function")
                return self._tool_calls_from_response(response)
            except Exception as error:
                if attempt >= self.retries:
                    raise LLMWrapperError("call_function", error) from error
                attempt += 1
                self._sleep_before_retry("call_function", error, attempt)

    def _tool_calls_from_response(self, response):
        tool_calls = response.get("tool_calls") or []
        parsed = []
        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            if function is not None:
                parsed.append(
                    {
                        "id": getattr(tool_call, "id", None),
                        "name": function.name,
                        "arguments": function.arguments,
                    }
                )
                continue
            parsed.append(
                {
                    "id": tool_call.get("id"),
                    "name": tool_call.get("function", {}).get("name"),
                    "arguments": tool_call.get("function", {}).get("arguments", "{}"),
                }
            )
        return parsed
