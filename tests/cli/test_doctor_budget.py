"""`agent-adapter doctor` rows for the shared budget stores."""

from __future__ import annotations

from agent_adapter.cli.doctor_features import FEATURES, check_feature

EXTRAS = {"redis": [], "sqlalchemy": []}


def row(topic: str):
    return next(feature for feature in FEATURES if feature.topic == topic)


def test_budget_redis_is_ready_or_names_the_extra():
    redis = row("budget redis")
    assert check_feature(redis, EXTRAS, {"redis": "8.1.0"}.get).level == "ok"
    absent = check_feature(redis, EXTRAS, {}.get)
    assert absent.level == "info" and "agent-adapter[redis]" in absent.message


def test_budget_sqlalchemy_needs_an_async_driver():
    sql = row("budget sqlalchemy")
    ready = check_feature(sql, EXTRAS, {"sqlalchemy": "2.0.40", "aiosqlite": "0.21"}.get)
    assert ready.level == "ok" and "aiosqlite 0.21" in ready.message
    no_driver = check_feature(sql, EXTRAS, {"sqlalchemy": "2.0.40"}.get)
    assert no_driver.level == "warn" and "asyncpg" in no_driver.message
