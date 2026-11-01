"""A strict, type-tagged canonical encoding: it seals flow checkpoints and keys the replay journal.

JSON natives stay as they are; every other supported type becomes a one-key `{"$tag": ...}`
object, so two different values never share an encoding (bytes vs base64 text, tuple vs list,
`1` vs `1.0` vs `True`). Plain dicts need str keys that do not start with "$"; any other dict
is a tagged, sorted pair list. Unknown types go through an optional encoder, else fail.
"""

from __future__ import annotations

import base64
import datetime as dt
import enum
import hashlib
import json
import math
import uuid
from collections.abc import Callable, Iterator, Mapping
from decimal import Decimal
from typing import Any, get_args
from zoneinfo import ZoneInfo

from pydantic import BaseModel, RootModel

from agent_adapter.core.errors import ToolUsageError

Encoder = Callable[[object], object]  # an unknown value -> a value this module can encode
Decoder = Callable[[str, object], object]  # (type "module:QualName", encoded value) -> value

# Ints past this many bits are tagged as hex: their decimal text could exceed
# sys.int_max_str_digits (640 digits at the lowest setting), which json.dumps refuses.
_SAFE_INT_BITS = 1_990
_MAX_INT_HEX = 1024 * 1024


class NotSealable(ToolUsageError):
    """A value has no canonical encoding; `path` names where, `type_name` what."""

    def __init__(self, path: str, type_name: str, reason: str = "") -> None:
        self.path = path
        self.type_name = type_name
        detail = f" ({reason})" if reason else ""
        super().__init__(f"value at {path} of type {type_name} cannot be sealed{detail}")


def type_name(cls: type) -> str:
    """`module:QualName`, the form `decode` looks classes up by."""
    return f"{cls.__module__}:{cls.__qualname__}"


def encode(value: Any, encoder: Encoder | None = None, path: str = "$") -> Any:
    """`value` in the tagged canonical form (JSON data).

    Raises:
        NotSealable: a value of an unknown type and no encoder (or the encoder failed).
    """
    try:
        return _encode(value, encoder, path)
    except RecursionError as exc:
        raise NotSealable(path, type(value).__qualname__, "nested too deeply") from exc


def _encode(value: Any, encoder: Encoder | None, path: str) -> Any:
    kind = type(value)
    if value is None or kind in (str, bool):
        return value
    if kind is int:
        return value if value.bit_length() <= _SAFE_INT_BITS else _encode_big_int(value, path)
    if kind is float:
        return value if math.isfinite(value) else {"$float": repr(value)}
    if kind is list:
        return [_encode(v, encoder, f"{path}[{i}]") for i, v in enumerate(value)]
    if kind is dict:
        return _encode_dict(value, encoder, path)
    if isinstance(value, enum.Enum):
        return {"$enum": type_name(kind), "value": _encode(value.value, encoder, path)}
    if isinstance(value, BaseModel):
        return _encode_model(value, encoder, path)
    if kind in (dt.datetime, dt.time):
        return _encode_moment(value, path)
    simple = _SIMPLE.get(kind)
    if simple is not None:
        tag, dump = simple
        return {tag: dump(value)}
    if kind in (tuple, set, frozenset):
        items = [_encode(v, encoder, f"{path}[{i}]") for i, v in enumerate(value)]
        if kind is not tuple:
            items.sort(key=canonical)
        return {f"${kind.__name__}": items}
    return _encode_custom(value, encoder, path)


def _encode_big_int(value: int, path: str) -> Any:
    text = format(value, "x")  # power-of-two bases have no text-length limit
    if len(text) > _MAX_INT_HEX:
        raise NotSealable(path, "int", "more than 1 MiB of hex digits")
    return {"$int": text}


def _encode_moment(value: dt.datetime | dt.time, path: str) -> Any:
    naive = value.replace(tzinfo=None, fold=0).isoformat()
    return {f"${type(value).__name__}": [naive, _encode_tz(value.tzinfo, path), value.fold]}


def _encode_tz(tz: dt.tzinfo | None, path: str) -> Any:
    if tz is None:
        return None
    if type(tz) is dt.timezone:
        offset = tz.utcoffset(None)
        encoded: dict[str, Any] = {"offset": offset.days * 86400 + offset.seconds}
        if offset.microseconds:
            encoded["us"] = offset.microseconds
        name = tz.tzname(None)
        if name != dt.timezone(offset).tzname(None):
            encoded["name"] = name
        return encoded
    if type(tz) is ZoneInfo:
        if tz.key is None:
            raise NotSealable(path, "ZoneInfo", "a ZoneInfo without a key (from_file)")
        return {"zone": tz.key}
    raise NotSealable(
        path, type(tz).__qualname__, "tzinfo must be datetime.timezone or zoneinfo.ZoneInfo"
    )


def _encode_dict(value: dict[Any, Any], encoder: Encoder | None, path: str) -> Any:
    if all(type(k) is str and not k.startswith("$") for k in value):
        return {k: _encode(v, encoder, f"{path}.{k}") for k, v in value.items()}
    pairs = [
        [_encode(k, encoder, f"{path}<key>"), _encode(v, encoder, f"{path}[{k!r}]")]
        for k, v in value.items()
    ]
    pairs.sort(key=lambda pair: canonical(pair[0]))
    return {"$dict": pairs}


def _encode_model(model: BaseModel, encoder: Encoder | None, path: str) -> Any:
    fields = {
        name: _encode(getattr(model, name), encoder, f"{path}.{name}")
        for name in type(model).model_fields
    }
    encoded: dict[str, Any] = {"$model": type_name(type(model)), "fields": fields}
    if model.__pydantic_extra__:
        encoded["extra"] = _encode(dict(model.__pydantic_extra__), encoder, f"{path}<extra>")
    return encoded


def _encode_custom(value: Any, encoder: Encoder | None, path: str) -> Any:
    name = type(value).__qualname__
    if encoder is None:
        raise NotSealable(path, name, "pass FlowRunner(seal_encoder=...) for custom types")
    try:
        replacement = encoder(value)
    except Exception as exc:
        raise NotSealable(path, name, f"seal_encoder raised {type(exc).__name__}") from exc
    if type(replacement) is type(value):
        raise NotSealable(path, name, "seal_encoder returned the same type")
    return {"$custom": type_name(type(value)), "value": _encode(replacement, encoder, path)}


_SIMPLE: dict[type, tuple[str, Callable[[Any], Any]]] = {
    bytes: ("$bytes", lambda v: base64.b64encode(v).decode("ascii")),
    bytearray: ("$bytearray", lambda v: base64.b64encode(bytes(v)).decode("ascii")),
    dt.date: ("$date", lambda v: v.isoformat()),
    dt.timedelta: ("$timedelta", lambda v: [v.days, v.seconds, v.microseconds]),
    Decimal: ("$decimal", str),
    uuid.UUID: ("$uuid", str),
}


def canonical(data: Any) -> str:
    """The canonical JSON text of encoded data: sorted keys, compact, non-ASCII kept.

    Raises:
        NotSealable: JSON cannot write the data (an untagged huge int, nesting too deep).
    """
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (ValueError, RecursionError) as exc:
        raise NotSealable("$", type(data).__qualname__, f"no JSON text: {exc}") from exc


def canonical_bytes(data: Any) -> bytes:
    """`canonical(data)` as UTF-8 (lone surrogates stay distinct)."""
    return canonical(data).encode("utf-8", "surrogatepass")


def digest(value: Any, encoder: Encoder | None = None) -> str:
    """SHA-256 hex of the canonical encoding of `value`.

    Raises:
        NotSealable: as `encode`.
    """
    return hashlib.sha256(canonical_bytes(encode(value, encoder))).hexdigest()


def check_plain(value: Any, path: str = "$") -> None:
    """Refuse anything but exact JSON types (encoded data read back from elsewhere).

    Raises:
        NotSealable: a value is not plain JSON.
    """
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return
    if kind is float and math.isfinite(value):
        return
    if kind is list:
        for i, item in enumerate(value):
            check_plain(item, f"{path}[{i}]")
        return
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise NotSealable(path, type(key).__qualname__, "non-string key")
            check_plain(item, f"{path}.{key}")
        return
    raise NotSealable(path, kind.__qualname__, "not plain JSON")


def decode(
    data: Any,
    decoder: Decoder | None = None,
    known: Mapping[str, type] | None = None,
) -> Any:
    """The value `encode` made `data` from; only call it on data whose MAC was checked.

    Args:
        known: The classes a `$model` / `$enum` may name, by `type_name` (`reachable_types`);
            any other name goes to `decoder`, else is refused before anything is built.

    Raises:
        NotSealable: unknown tag, a class not allowed, or a value the decoder refused.
    """
    if isinstance(data, list):
        return [decode(v, decoder, known) for v in data]
    if not isinstance(data, dict):
        return data
    tag = next((k for k in data if k.startswith("$")), None)
    if tag is None:
        return {k: decode(v, decoder, known) for k, v in data.items()}
    return _decode_tagged(tag, data, decoder, known or {})


def _decode_tagged(
    tag: str, data: dict[str, Any], decoder: Decoder | None, known: Mapping[str, type]
) -> Any:
    value = data[tag]
    plain = _UNDUMP.get(tag)
    if plain is not None:
        return plain(value)
    if tag in ("$tuple", "$set", "$frozenset"):
        items = [decode(v, decoder, known) for v in value]
        return {"$tuple": tuple, "$set": set, "$frozenset": frozenset}[tag](items)
    if tag == "$dict":
        return {decode(k, decoder, known): decode(v, decoder, known) for k, v in value}
    if tag == "$enum":
        cls = _allowed(value, known, enum.Enum, decoder)
        member = decode(data["value"], decoder, known)
        return cls(member) if cls is not None else _delegate(decoder, value, member)
    if tag == "$model":
        model = _allowed(value, known, BaseModel, decoder)
        fields = decode(data["fields"], decoder, known)
        fields.update(decode(data.get("extra", {}), decoder, known))
        if model is None:
            return _delegate(decoder, value, fields)
        return model.model_construct(**fields)
    if tag == "$custom":
        return _delegate(decoder, value, decode(data["value"], decoder, known))
    raise NotSealable("$", tag, "unknown tag")


def _decode_int(value: Any) -> int:
    if not isinstance(value, str):
        raise NotSealable("$", type(value).__qualname__, "an $int must be hex text")
    return int(value, 16)


def _decode_moment(tag: str, value: Any) -> dt.datetime | dt.time:
    text, tz, fold = value
    if not isinstance(text, str) or type(fold) is not int or fold not in (0, 1):
        raise NotSealable("$", tag, "malformed date or time")
    kind = dt.datetime if tag == "$datetime" else dt.time
    return kind.fromisoformat(text).replace(tzinfo=_decode_tz(tz), fold=fold)


def _decode_tz(tz: Any) -> dt.tzinfo | None:
    if tz is None:
        return None
    if "zone" in tz:
        return ZoneInfo(str(tz["zone"]))
    offset = dt.timedelta(seconds=tz["offset"], microseconds=tz.get("us", 0))
    return dt.timezone(offset, tz["name"]) if "name" in tz else dt.timezone(offset)


_UNDUMP: dict[str, Callable[[Any], Any]] = {
    "$int": _decode_int,
    "$float": float,
    "$bytes": base64.b64decode,
    "$bytearray": lambda v: bytearray(base64.b64decode(v)),
    "$datetime": lambda v: _decode_moment("$datetime", v),
    "$date": dt.date.fromisoformat,
    "$time": lambda v: _decode_moment("$time", v),
    "$timedelta": lambda v: dt.timedelta(days=v[0], seconds=v[1], microseconds=v[2]),
    "$decimal": Decimal,
    "$uuid": uuid.UUID,
}


def _allowed(name: str, known: Mapping[str, type], base: type, decoder: Decoder | None) -> Any:
    """The allowed class `name`, or None when the decoder takes it; checked before any build."""
    cls = known.get(name)
    if cls is None:
        if decoder is None:
            raise NotSealable(
                "$",
                name,
                "class not allowed: not reachable from the state model's annotations "
                "(a seal_decoder may load it)",
            )
        return None
    if not isinstance(cls, type) or not issubclass(cls, base):
        raise NotSealable("$", name, f"not a {base.__name__}")
    return cls


def _delegate(decoder: Decoder | None, name: str, value: Any) -> Any:
    if decoder is None:
        raise NotSealable("$", name, "pass FlowRunner(seal_decoder=...) to load it")
    try:
        return decoder(name, value)
    except Exception as exc:
        raise NotSealable("$", name, f"seal_decoder refused it ({type(exc).__name__})") from exc


# library bases: their subclasses are every model / enum loaded, never an allow-list
_ROOTS: frozenset[type] = frozenset(
    {
        BaseModel,
        RootModel,
        enum.Enum,
        enum.ReprEnum,
        enum.IntEnum,
        enum.StrEnum,
        enum.Flag,
        enum.IntFlag,
    }
)


def _library_base(cls: type) -> bool:
    """A pydantic or enum class: expanding its subclasses would allow every loaded one."""
    module = cls.__module__ or ""
    return cls in _ROOTS or module == "enum" or module.startswith("pydantic")


def reachable_types(*annotations: Any) -> dict[str, type]:
    """Models and enums reachable from type annotations, by name.

    Follows fields, unions and containers, and the already-loaded subclasses of each model
    or enum found (a `Dog` in an `Animal` field); nothing is imported.
    """
    found: dict[str, type] = {}
    todo = list(annotations)
    while todo:
        annotation = todo.pop()
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum | BaseModel):
            name = type_name(annotation)
            if name in found:
                continue
            found[name] = annotation
            if not _library_base(annotation):
                todo.extend(annotation.__subclasses__())
            if issubclass(annotation, BaseModel):
                todo.extend(field.annotation for field in annotation.model_fields.values())
        todo.extend(get_args(annotation))
    return found


def tagged_classes(data: Any) -> set[str]:
    """The `$model` / `$enum` class names in encoded data."""
    return set(_tagged(data))


def _tagged(data: Any) -> Iterator[str]:
    if isinstance(data, list):
        for item in data:
            yield from _tagged(item)
    elif isinstance(data, dict):
        for tag in ("$model", "$enum"):
            if isinstance(data.get(tag), str):
                yield data[tag]
        for item in data.values():
            yield from _tagged(item)
