import os

import pytest


@pytest.mark.integration
def test_env_variables():
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_TEST_MODEL")
    base_url = os.environ.get("OPENAI_BASE_URL")

    if not api_key or not model or not base_url:
        pytest.skip("Set OPENAI_API_KEY, OPENAI_TEST_MODEL, and OPENAI_BASE_URL to run this integration test.")

    assert api_key is not None, "OPENAI_API_KEY environment variable is not set."
    assert model is not None, "OPENAI_TEST_MODEL environment variable is not set."
    assert base_url is not None, "OPENAI_BASE_URL environment variable is not set."
