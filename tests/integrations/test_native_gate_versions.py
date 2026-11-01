"""`check_framework` compares as many release parts as the range gives (patch-level floors)."""

import pytest

from agent_adapter.core.errors import IntegrationError
from agent_adapter.integrations._native_gate import check_framework

PATCH_RANGE = ((0, 2, 150), (0, 3, 0))


@pytest.mark.parametrize("version", ["0.2.150", "0.2.175", "0.2.999", "0.2.150.post1"])
def test_check_framework_accepts_a_release_inside_a_patch_range(version: str) -> None:
    check_framework("sdk", version, PATCH_RANGE, {})


@pytest.mark.parametrize("version", ["0.2.149", "0.2", "0.3.0", "1.0.0", "dev", ""])
def test_check_framework_refuses_a_release_outside_a_patch_range(version: str) -> None:
    with pytest.raises(
        IntegrationError, match=r"outside the supported range 0\.2\.150 to below 0\.3\.0"
    ):
        check_framework("sdk", version, PATCH_RANGE, {})


def test_check_framework_reads_the_leading_digits_of_each_part() -> None:
    check_framework("fw", "1.23rc1", ((1, 23), (2, 0)), {})
    for version in ("1.22.99", "1.x"):
        with pytest.raises(IntegrationError, match=r"range 1\.23 to below 2\.0"):
            check_framework("fw", version, ((1, 23), (2, 0)), {})
