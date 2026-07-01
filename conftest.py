import pytest
from typing import Any


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--bonus-encoder", action="store_true", default=False)


@pytest.fixture(scope="session")
def bonus_encoder(request: pytest.FixtureRequest) -> Any:
    return request.config.getoption("--bonus-encoder")
