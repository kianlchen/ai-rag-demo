import pytest

from app.rag.store import STORE
from app.summarize.config import settings


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--llm-provider",
        action="store",
        choices=("dummy", "openai"),
        help="Select LLM provider for tests (default: dummy)",
        default="dummy",
    )


@pytest.fixture(autouse=True, scope="session")
def _set_llm_provider(request: pytest.FixtureRequest):
    """Default to DummyLLM unless pytest run uses --llm-provider=openai"""
    if request.config.getoption("--llm-provider") == "dummy":
        settings.llm_provider = "dummy"
        settings.openai_api_key = None


@pytest.fixture(autouse=True)
def _clear_store():
    """Isolate the in-memory RAG store per test"""
    STORE.clear()
    yield
    STORE.clear()
