import pytest

import lucem


@pytest.fixture(autouse=True, scope="session")
def cpu():
    # The tests run on the CPU runtime (CI has no GPU); results do not depend on the runtime
    lucem.runtime.use("cpu")
