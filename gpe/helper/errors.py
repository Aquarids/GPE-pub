class SearchProviderError(RuntimeError):
    def __init__(self, provider, message, original_error=None, **context):
        self.provider = provider
        self.original_error = original_error
        self.context = {key: value for key, value in context.items() if value is not None}
        detail = f"{provider}: {message}"
        if original_error is not None:
            detail = f"{detail}: {type(original_error).__name__}: {original_error}"
        if self.context:
            context_text = ", ".join(f"{key}={value}" for key, value in self.context.items())
            detail = f"{detail} ({context_text})"
        super().__init__(detail)
