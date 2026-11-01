"""Hello world: one function, one call, through the default executor."""

import agent_adapter as aa


@aa.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


print(aa.call(add, a=1, b=2))  # 3: validated and guarded by the default executor
