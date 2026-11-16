"""Media in tool results: images, audio, documents and files a tool returns to the model."""

from __future__ import annotations

import base64
import binascii
import mimetypes
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator

MediaKind = Literal["image", "audio", "video", "document", "file"]

# The built-in tables only: the module-level guess also reads the Windows registry (.csv -> Excel).
_TYPES = mimetypes.MimeTypes()

MAX_MEDIA_BYTES = 20 * 1024 * 1024
"""The largest `Media` payload accepted by default; providers apply their own lower limits."""

_DOCUMENTS = {
    "application/pdf",
    "application/json",
    "application/msword",
    "application/rtf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
# (offset, magic, MIME). RIFF and ftyp containers are told apart by a second check below.
_MAGIC: tuple[tuple[int, bytes, str], ...] = (
    (0, b"\x89PNG\r\n\x1a\n", "image/png"),
    (0, b"\xff\xd8\xff", "image/jpeg"),
    (0, b"GIF87a", "image/gif"),
    (0, b"GIF89a", "image/gif"),
    (0, b"%PDF-", "application/pdf"),
    (0, b"ID3", "audio/mpeg"),
    (0, b"\xff\xfb", "audio/mpeg"),
    (0, b"\xff\xf3", "audio/mpeg"),
    (0, b"OggS", "audio/ogg"),
    (0, b"fLaC", "audio/flac"),
    (0, b"\x1a\x45\xdf\xa3", "video/webm"),
    (0, b"BM", "image/bmp"),
)


def sniff_mime(data: bytes) -> str:
    """The MIME type the leading bytes of `data` announce, or "" when unknown."""
    for offset, magic, mime in _MAGIC:
        if data[offset : offset + len(magic)] == magic:
            return mime
    if data[:4] == b"RIFF":
        return {b"WEBP": "image/webp", b"WAVE": "audio/wav"}.get(data[8:12], "")
    if data[4:8] == b"ftyp":
        return "audio/mp4" if data[8:11] == b"M4A" else "video/mp4"
    head = data[:256].lstrip().lower()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head):
        return "image/svg+xml"
    return ""


def media_kind(mime_type: str) -> MediaKind:
    """The broad kind of a MIME type."""
    major = mime_type.split("/", 1)[0]
    if major in ("image", "audio", "video"):
        return major  # type: ignore[return-value]
    if mime_type in _DOCUMENTS or major == "text":
        return "document"
    return "file"


class Media(BaseModel):
    """An image, audio clip, document or file a tool returns, as bytes or a URL.

    Return it alone, or in a list with text and JSON values. A missing `mime_type` is
    sniffed from the bytes, else guessed from `name` or `url`. A `str` for `data` is
    read as base64.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["media"] = "media"  # marks the dict form, so a cached result revives
    mime_type: str = ""
    data: bytes = b""
    url: str = ""
    name: str = ""

    @field_validator("data", mode="before")
    @classmethod
    def _base64(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        altchars = b"-_" if "-" in value or "_" in value else None  # URL-safe base64 too
        try:
            return base64.b64decode(value, altchars=altchars, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("data as a str must be base64") from exc

    @field_serializer("data", when_used="json")
    def _standard_base64(self, data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")  # pydantic's own JSON bytes are URL-safe

    @model_validator(mode="after")
    def _check(self) -> Media:
        if bool(self.data) == bool(self.url):
            raise ValueError("Media needs exactly one of data or url")
        if len(self.data) > MAX_MEDIA_BYTES:
            raise ValueError(f"Media is {len(self.data)} bytes; the limit is {MAX_MEDIA_BYTES}")
        if not self.mime_type:
            guess = sniff_mime(self.data) if self.data else ""
            guess = guess or _TYPES.guess_type(self.name or self.url)[0] or ""
            object.__setattr__(self, "mime_type", guess or "application/octet-stream")
        return self

    @classmethod
    def from_file(
        cls, path: str | Path, *, mime_type: str = "", max_bytes: int = MAX_MEDIA_BYTES
    ) -> Media:
        """Read a local file; raises ValueError when it is larger than `max_bytes`."""
        file = Path(path)
        size = file.stat().st_size
        if size > max_bytes:
            raise ValueError(f"{file.name} is {size} bytes; the limit is {max_bytes}")
        return cls(data=file.read_bytes(), mime_type=mime_type, name=file.name)

    @property
    def kind(self) -> MediaKind:
        """image, audio, video, document or file."""
        return media_kind(self.mime_type)

    @property
    def label(self) -> str:
        """A name for the model to refer to: `name`, the URL's last segment, or the kind."""
        if self.name:
            return self.name
        tail = self.url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
        return tail or self.kind

    def base64(self) -> str:
        """The bytes as base64 text ("" for a URL)."""
        return base64.b64encode(self.data).decode("ascii")

    def data_url(self) -> str:
        """A `data:` URL for the bytes, or the URL itself."""
        return self.url or f"data:{self.mime_type};base64,{self.base64()}"

    def describe(self) -> str:
        """A short description, e.g. `image/png 'chart.png' (34 KB)`."""
        size = f"{max(1, round(len(self.data) / 1024))} KB" if self.data else self.url
        return f"{self.mime_type} {self.label!r} ({size})"


def media_note(media: Media, reason: str) -> str:
    """The text a model sees in place of media it cannot take: what was left out and why."""
    return f"[{media.kind} not shown: {media.describe()}; {reason}]"


def as_media(value: Any) -> Media | None:
    """`value` as Media: a Media, or its dict form (e.g. back from a cache); else None."""
    if isinstance(value, Media):
        return value
    if isinstance(value, dict) and value.get("type") == "media":
        try:
            return Media.model_validate(value)
        except ValueError:
            return None
    return None


def split_media(value: Any) -> tuple[Any, tuple[Media, ...]]:
    """A tool value without its media, and the media: from a Media or a list holding some.

    Only the value itself and the items of a top-level list or tuple are searched. The rest
    is None when nothing but media was returned, the one item left, or the list left.
    """
    single = as_media(value)
    if single is not None:
        return None, (single,)
    if not isinstance(value, list | tuple):
        return value, ()
    media: list[Media] = []
    rest: list[Any] = []
    for item in value:
        found = as_media(item)
        if found is None:
            rest.append(item)
        else:
            media.append(found)
    if not media:
        return value, ()
    return (None if not rest else rest[0] if len(rest) == 1 else rest), tuple(media)


def mentions_media(annotation: Any) -> bool:
    """Whether a type annotation is or holds `Media` (e.g. `list[str | Media]`)."""
    if annotation is Media:
        return True
    return any(mentions_media(arg) for arg in get_args(annotation))


def notes(media: Iterable[Media], reason: str) -> str:
    """One note line per media item."""
    return "\n".join(media_note(item, reason) for item in media)
