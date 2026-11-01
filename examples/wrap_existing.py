"""Expose existing code as tools without editing it."""

import asyncio
import functools
import json

from agent_adapter import Executor, Registry


# --8<-- [start:legacy]
# Existing code. None of it changes.
def get_articles(article_list: list[str], lang: str, debug: bool = False) -> dict[str, str]:
    """Return the text of help-center articles.

    Args:
        article_list: Article topics, e.g. ["returns", "shipping"].
        lang: Language code.
        debug: Verbose output.
    """
    return {article: f"Article {article} ({lang})" for article in article_list}


class RateService:
    """Shipping rates for one region."""

    def __init__(self, region: str) -> None:
        self.region = region

    def lookup(self, item: str) -> float:
        """Return the shipping rate for an item, in percent of the order value.

        Args:
            item: Item name.
        """
        return 18.0 if self.region == "us" else 5.0


def convert(amount: float, rate: float) -> float:
    """Convert a USD amount with an exchange rate.

    Args:
        amount: Amount in USD.
        rate: Units of the target currency per USD.
    """
    return round(amount * rate, 2)


# --8<-- [end:legacy]

registry = Registry()

# --8<-- [start:wrap]
registry.wrap(
    get_articles,
    description="Full text of help-center articles.",
    fixed={"lang": "en"},  # bound value; the LLM never sees it
    hide="debug",  # hidden; its default applies
    rename={"article_list": "articles"},  # the LLM sees `articles`
    result=lambda texts: list(texts.values()),  # adapt the return shape
    parallel_safe=True,
)

# A bound method: register obj.method, not the method in the class body.
registry.wrap(RateService("us").lookup, name="shipping_rate")

# A partial: its keyword bindings act like fixed= values.
registry.wrap(functools.partial(convert, rate=0.79), name="usd_to_gbp")

# One function, several tools with different fixed values.
registry.wrap(convert, name="usd_to_eur", fixed={"rate": 0.92})
# --8<-- [end:wrap]


async def main() -> None:
    # --8<-- [start:call]
    spec = registry.get("get_articles")
    print(list(spec.json_schema()["properties"]))  # ['articles']
    print(dict(spec.llm_to_python))  # {'articles': 'article_list'}
    executor = Executor(registry)  # every call goes through the executor's policies
    articles = await executor.run("get_articles", {"articles": ["returns"]})
    print(articles)  # ['Article returns (en)']
    print(await executor.run("shipping_rate", {"item": "laptop"}))  # 18.0
    print(await executor.run("usd_to_gbp", {"amount": 10}))  # 7.9
    print(await executor.run("usd_to_eur", {"amount": 10}))  # 9.2
    # --8<-- [end:call]
    print(json.dumps(registry.names()))


if __name__ == "__main__":
    asyncio.run(main())
