from .model_provider import ModelProvider
from .local_provider import LocalTransformersProvider
from .openai_provider import OpenAICompatibleProvider, VLLMProvider
from typing import Any

def create_provider(
    provider_config: dict[str, Any],
    max_new_tokens: int | None = None,
    temperature: float | None = None,
)-> ModelProvider:
    provider_type = provider_config['active']
    max_new_tokens = provider_config.get('max_new_token', max_new_tokens if max_new_tokens is not None else 512)
    temperature = provider_config.get('temperature', temperature if temperature is not None else 0.0)

    if provider_type == "local":
        return LocalTransformersProvider(
            model=provider_config['local']['model_path'],
            max_new_tokens=max_new_tokens or 512,
            temperature=temperature or 0.0,
            device=provider_config['local']['device'],
        )

    elif provider_type == "openai":
        return OpenAICompatibleProvider(
            model=provider_config['openai']['model'],
            base_url=provider_config['openai']['base_url'],
            api_key=provider_config['openai']['api_key'],
            max_new_tokens=max_new_tokens or 512,
            temperature=temperature or 0.0,
        )
    
    elif provider_type == "vllm":
        return VLLMProvider(
            model=provider_config['local']['model_path'],
            max_new_tokens=max_new_tokens or 512,
            temperature=temperature or 0.0,
        )

    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")

__all__ = [
    "LocalTransformersProvider",
    "OpenAICompatibleProvider",
    "VLLMProvider",
    "create_provider",
]
