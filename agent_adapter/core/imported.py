"""Build a ToolSpec from a raw JSON schema: tools imported from MCP, LangChain, OpenAPI."""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel

from agent_adapter.core._policy import validate_policy
from agent_adapter.core._signature import NAME_RE, args_model_name
from agent_adapter.core.errors import ToolDefinitionError
from agent_adapter.core.tool import ToolSpec
from agent_adapter.schema.models import field_aliases, is_object_schema, model_from_json_schema
from agent_adapter.schema.portable import check_portable

logger = logging.getLogger(__name__)

_BUILD_LOCK = threading.Lock()  # one argument model per imported tool, built once


class ImportedToolSpec(ToolSpec):
    """A `ToolSpec` from a raw schema whose argument model is built on first use.

    An import can register thousands of tools; each pydantic model is built only when the
    tool is validated, called or exported with its model, then cached.
    """

    @property
    def args_model(self) -> type[BaseModel]:
        """The loose argument model (`model_from_json_schema` of `input_schema`)."""
        model = self.__dict__.get("_args_model")
        if model is None:
            with _BUILD_LOCK:
                model = self.__dict__.get("_args_model")
                if model is None:
                    schema = self.input_schema or {}
                    model = model_from_json_schema(args_model_name(self.name), schema)
                    self.__dict__["_args_model"] = model
        return model

    @args_model.setter
    def args_model(self, model: type[BaseModel] | None) -> None:
        # The frozen dataclass __init__ sets every field once; None = build on first use.
        if model is not None:
            self.__dict__["_args_model"] = model


def spec_from_schema(
    name: str,
    description: str,
    input_schema: Mapping[str, Any],
    func: Callable[..., Any],
    *,
    output_schema: Mapping[str, Any] | None = None,
    portable: bool | None = None,
    context_param: str | None = None,
    **policy: Any,
) -> ToolSpec:
    """A tool whose arguments are described by `input_schema` instead of a signature.

    `func` receives the set arguments as keywords named exactly as the schema properties
    (plus `context_param=ctx` when set); unset optional arguments are not sent.

    Args:
        portable: None checks the schema and records the result; True makes `registry.add`
            reject a non-portable schema; False skips the check.
        **policy: Execution options as for `build_spec` (`kind`, `tags`, `effects`, ...).

    Raises:
        ToolDefinitionError: bad name, empty description, non-object schema or bad option.
    """
    _check_inputs(name, description, input_schema, func, portable, context_param)
    options = validate_policy(name, policy)
    schema = copy.deepcopy(dict(input_schema))
    if portable is None:
        problems = check_portable(schema)
        portable = not problems
        if problems:
            logger.debug("tool %s: schema is not portable: %s", name, "; ".join(problems))
    return ImportedToolSpec(
        name=name,
        description=description.strip(),
        args_model=None,  # type: ignore[arg-type]  # built on first use
        func=func,
        context_param=context_param,
        llm_to_python=field_aliases(schema),
        input_schema=schema,
        output_schema=copy.deepcopy(dict(output_schema)) if output_schema is not None else None,
        portable=portable,
        **options,
    )


def _check_inputs(
    name: Any,
    description: Any,
    input_schema: Any,
    func: Any,
    portable: Any,
    context_param: Any,
) -> None:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ToolDefinitionError(
            f"invalid tool name {name!r}: use letters, digits and '_' (max 64)"
        )
    if not isinstance(description, str) or not description.strip():
        raise ToolDefinitionError(f"tool {name!r} needs a non-empty description")
    if not is_object_schema(input_schema):
        raise ToolDefinitionError(f"tool {name!r}: input_schema root must be an object")
    if not callable(func):
        raise ToolDefinitionError(f"tool {name!r}: func must be callable, got {func!r}")
    if portable is not None and not isinstance(portable, bool):
        raise ToolDefinitionError(f"tool {name!r}: portable must be a bool or None")
    if context_param is not None:
        if not isinstance(context_param, str) or not context_param.isidentifier():
            raise ToolDefinitionError(f"tool {name!r}: context_param must be an identifier")
        if context_param in (input_schema.get("properties") or {}):
            raise ToolDefinitionError(
                f"tool {name!r}: context_param {context_param!r} is also a schema property"
            )
