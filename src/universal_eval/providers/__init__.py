from .model_provider import ModelProvider
from .local_provider import LocalTransformersProvider
from .openai_provider import OpenAICompatibleProvider, VLLMProvider
from typing import Any

def create_provider(
    provider_config: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
)-> ModelProvider:
    provider_type = provider_config['active']

    if provider_type == "local":
        return LocalTransformersProvider(
            model=provider_config['local']['model_path'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=provider_config['local']['device'],
        )

    elif provider_type == "openai":
        return OpenAICompatibleProvider(
            model=provider_config['openai']['model'],
            base_url=provider_config['openai']['base_url'],
            api_key=provider_config['openai']['api_key'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    
    elif provider_type == "vllm":
        return VLLMProvider(
            model=provider_config['local']['model_path'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")

__all__ = [
    "LocalTransformersProvider",
    "OpenAICompatibleProvider",
    "VLLMProvider",
    "create_provider",
]
