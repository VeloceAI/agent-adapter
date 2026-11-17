"""Offline vision check: an agent calls `look`, then answers from the frame with SmolVLM.

No API key or sign-in. In a separate venv (never add torch to the package):

    pip install -e . pillow av transformers num2words
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python scripts/vision_check.py

The first run downloads SmolVLM-256M-Instruct (Apache-2.0, about 500 MB) from Hugging Face.
Five drawn frames go through `VideoFeed`, `look_tool` and the tool loop; the script prints
each answer and how many of five name the frame correctly.
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import agent_adapter as aa
from agent_adapter.voice.video import VideoFeed, look_tool

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
SHAPE = "Describe the picture in one short sentence."  # a two-part question gets half an answer
WORD = "What word is written in the picture? Answer with the word."


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (512, 384), "white")
    return image, ImageDraw.Draw(image)


def circle(fill: str) -> Image.Image:
    image, draw = canvas()
    draw.ellipse((136, 72, 376, 312), fill=fill)
    return image


def square(fill: str) -> Image.Image:
    image, draw = canvas()
    draw.rectangle((136, 72, 376, 312), fill=fill)
    return image


def triangle(fill: str) -> Image.Image:
    image, draw = canvas()
    draw.polygon([(256, 52), (416, 332), (96, 332)], fill=fill)
    return image


def word(text: str) -> Image.Image:
    image, draw = canvas()
    font = ImageFont.load_default(size=140)
    draw.text((256, 192), text, fill="black", font=font, anchor="mm")
    return image


# (frame, question, words that must all appear in the answer: alternatives split by "|")
CASES: list[tuple[Image.Image, str, list[str]]] = [
    (circle("red"), SHAPE, ["red", "circle|round|ball|dot|disc"]),
    (square("blue"), SHAPE, ["blue", "square|rectangle|box|cube"]),
    (triangle("green"), SHAPE, ["green", "triangle|triangular|pyramid"]),
    (circle("yellow"), SHAPE, ["yellow", "circle|round|ball|dot|disc|sun"]),
    (word("STOP"), WORD, ["stop"]),
]


class SmolVLM:
    """A `CallableModel` function: asks for `look`, then answers from the returned image."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor: Any = AutoProcessor.from_pretrained(
            MODEL_ID,
            size={"longest_edge": 512},  # fewer image tiles: ~10 s an answer on CPU
        )
        self.model: Any = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)

    def __call__(self, request: aa.LLMRequest) -> dict[str, Any]:
        last = request.messages[-1]
        if last.role != "tool":
            return {"tool_calls": [{"name": "look", "arguments": {}}]}
        if not last.media:
            return {"text": f"I could not see anything: {last.content}"}
        question = next(m.content for m in request.messages if m.role == "user")
        image = Image.open(io.BytesIO(last.media[0].data)).convert("RGB")
        return {"text": self.answer(image, str(question))}

    def answer(self, image: Image.Image, question: str) -> str:
        chat = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = self.processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=24, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :])[0]
        return str(reply).replace("<end_of_utterance>", "").strip()


def correct(answer: str, expected: list[str]) -> bool:
    text = answer.lower()
    return all(any(option in text for option in group.split("|")) for group in expected)


async def main() -> int:
    feed = VideoFeed(max_side=512)
    registry = aa.Registry()
    registry.add(look_tool(feed))
    executor = aa.Executor(registry)
    model = aa.CallableModel(SmolVLM(), provider="local", model="smolvlm-256m")
    score = 0
    for frame, question, expected in CASES:
        feed.clear()
        feed.push(frame)  # a PIL image; `look` encodes it to JPEG
        started = time.perf_counter()
        ask = [aa.Message(role="user", content=question)]
        result = await aa.run_tool_loop(model, registry, ask, executor=executor)
        answer = result.final.text or ""
        ok = correct(answer, expected)
        score += ok
        seconds = time.perf_counter() - started
        print(f"{'PASS' if ok else 'FAIL'}  {expected}  {answer!r}  ({seconds:.1f} s)")
    print(f"score: {score} of {len(CASES)}")
    return 0 if score == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
