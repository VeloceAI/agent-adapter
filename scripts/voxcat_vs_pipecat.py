"""Pipecat 1.16 side of voxcat_vs.py: its own voice pipeline with fake STT, LLM, TTS and VAD.

input -> STT -> user aggregator (VAD + turn strategies) -> LLM -> TTS -> output -> assistant
aggregator, on the real clock. Run it in the bench venv to print one trial per setup/scenario.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from functools import cache
from typing import Any

from voxcat_vs_common import (
    FRAME_BYTES,
    RATE,
    SCENARIOS,
    SETUPS,
    Counters,
    Delays,
    Playout,
    Scenario,
    Setup,
    Trial,
    caller_frames,
    heard,
    is_speech,
    reply_tokens,
    score,
    send_caller,
    tts_audio,
)

FLUX_START_MS = 200.0  # speech before the fake Flux says StartOfTurn, like VAD's start_secs


@cache
def _pipecat() -> Callable[..., Any]:
    # Pipecat logs a banner on import, so loguru is quieted before the first pipecat import.
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    import numpy as np
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
    from pipecat.frames import frames as f
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.llm_service import LLMService
    from pipecat.services.settings import LLMSettings, STTSettings, TTSSettings
    from pipecat.services.stt_latency import DEEPGRAM_TTFS_P99
    from pipecat.services.stt_service import STTService
    from pipecat.services.tts_service import TTSService
    from pipecat.transports.base_input import BaseInputTransport
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import TransportParams
    from pipecat.turns.eager_end_of_turn_mixin import EagerEndOfTurnSTTServiceMixin
    from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy
    from pipecat.turns.user_start import VADUserTurnStartStrategy as VADStart
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.utils.time import time_now_iso8601
    from pipecat.workers.runner import WorkerRunner

    class EnergyVAD(VADAnalyzer):
        def num_frames_required(self) -> int:
            return 512 if self.sample_rate == 16000 else 256  # Silero's window

        def voice_confidence(self, buffer: bytes) -> float:
            return 1.0 if is_speech(buffer) else 0.0

    class OracleSmartTurn(LocalSmartTurnAnalyzerV3):
        """Runs the real bundled model (its cost stays), but on our tones it always says
        'incomplete', so the verdict is replaced by the truth: has the caller finished?"""

        def __init__(self, done: asyncio.Event) -> None:
            super().__init__()
            self._done = done

        def _predict_endpoint(self, audio_array: Any) -> dict[str, Any]:
            result = super()._predict_endpoint(audio_array)
            return {"prediction": int(self._done.is_set()), "probability": result["probability"]}

        def warm(self) -> None:
            self.set_sample_rate(RATE)
            LocalSmartTurnAnalyzerV3._predict_endpoint(self, np.zeros(RATE, dtype=np.float32))

    class CallerInput(BaseInputTransport):
        def __init__(self, params: TransportParams) -> None:
            super().__init__(params)
            self.ready = asyncio.Event()

        async def start(self, frame: Any) -> None:
            await super().start(frame)
            await self.set_transport_ready(frame)
            self.ready.set()

        async def push(self, pcm: bytes) -> None:
            await self.push_audio_frame(
                f.InputAudioRawFrame(audio=pcm, sample_rate=RATE, num_channels=1)
            )

    class Speaker(BaseOutputTransport):
        def __init__(self, params: TransportParams, playout: Playout) -> None:
            super().__init__(params)
            self._playout = playout

        async def start(self, frame: Any) -> None:
            await super().start(frame)
            await self.set_transport_ready(frame)

        async def write_audio_frame(self, frame: Any) -> bool:
            await self._playout.play(frame.audio)
            return True

    class FakeSTT(EagerEndOfTurnSTTServiceMixin, STTService):
        """Streaming STT, Deepgram-like (finalize on VAD stop) or, with `flux`, Flux-like:
        the STT itself proposes turn start/end and an eager end, as DeepgramFluxSTTBase does."""

        def __init__(self, scenario: Scenario, delays: Delays, flux: bool) -> None:
            super().__init__(
                enable_eager_end_of_turn=flux,
                ttfs_p99_latency=DEEPGRAM_TTFS_P99,
                sample_rate=RATE,
                settings=STTSettings(model=None, language=None),
            )
            self._scenario, self._delays, self._flux = scenario, delays, flux
            self._speech = self._quiet = self._run = 0.0
            self._sent_words = 0
            self._pending = False  # speech heard that no final transcript covers yet
            self._finalizing: asyncio.Task | None = None
            self._in_turn = self._eager_sent = False
            self._last_interim = ""

        @property
        def supports_ttfs(self) -> bool:
            return not self._flux

        def service_metadata_frame(self) -> Any:
            frame = super().service_metadata_frame()
            if self._flux:
                frame.user_turn_strategies = self.recommended_user_turn_strategies(
                    enable_interruptions=True
                )
            return frame

        def _words(self) -> list[str]:
            return heard(self._scenario, self._speech).split()

        def _segment(self) -> str:
            words = self._words()
            text, self._sent_words = " ".join(words[self._sent_words :]), len(words)
            return text

        def _final(self, text: str, finalized: bool) -> Any:
            return f.TranscriptionFrame(text, "", time_now_iso8601(), finalized=finalized)

        async def run_stt(self, audio: bytes) -> Any:
            if is_speech(audio):
                self._speech, self._quiet, self._run = self._speech + 20, 0.0, self._run + 20
                self._pending = True
                if self._flux:
                    await self._flux_speech()
                text = " ".join(self._words()[self._sent_words :])
                if text and text != self._last_interim:
                    self._last_interim = text
                    yield f.InterimTranscriptionFrame(text, "", time_now_iso8601())
                return
            self._quiet, self._run = self._quiet + 20, 0.0
            if self._flux:
                await self._flux_quiet()
            elif (
                self._pending
                and not self._finalizing
                and self._quiet >= self._delays.stt_endpoint_ms
            ):
                self._pending = False
                if text := self._segment():  # Deepgram sends no empty finals
                    yield self._final(text, finalized=False)

        async def _flux_speech(self) -> None:
            if self._eager_sent:
                self._eager_sent = False
                await self._cancel_eager_end_of_turn()  # TurnResumed
            if not self._in_turn and self._run >= FLUX_START_MS:
                self._in_turn = True
                await self.broadcast_frame(f.ProposedUserStartedSpeakingFrame)

        async def _flux_quiet(self) -> None:
            if not self._in_turn:
                return
            if not self._eager_sent and self._quiet >= self._delays.flux_eager_ms:
                self._eager_sent = True
                text = " ".join(self._words()[self._sent_words :])
                await self._push_eager_end_of_turn(text, user_id="")
            if self._quiet >= self._delays.flux_end_ms:
                self._in_turn = self._eager_sent = self._pending = False
                self._clear_eager_end_of_turn()
                await self.push_frame(self._final(self._segment(), finalized=True))
                await self.broadcast_frame(f.ProposedUserStoppedSpeakingFrame)

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            stopped = isinstance(frame, f.VADUserStoppedSpeakingFrame)
            if stopped and not self._flux and self._pending and not self._finalizing:
                self._finalizing = self.create_task(self._finalize(), name="fake_finalize")

        async def _finalize(self) -> None:
            await asyncio.sleep(self._delays.stt_ms / 1000)
            self._finalizing = None
            if self._pending and (text := self._segment()):
                self._pending = False
                await self.push_frame(self._final(text, finalized=True))

    class FakeLLM(LLMService):
        def __init__(self, delays: Delays, counters: Counters) -> None:
            none = dict.fromkeys(("model", "system_instruction", "temperature", "max_tokens"))
            none |= dict.fromkeys(("top_p", "top_k", "frequency_penalty", "presence_penalty"))
            none |= {"seed": None, "user_turn_completion_config": None}
            super().__init__(settings=LLMSettings(filter_incomplete_user_turns=False, **none))
            self._delays, self._counters = delays, counters

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            if not isinstance(frame, f.LLMContextFrame):
                await self.push_frame(frame, direction)
                return
            await self.push_frame(f.LLMFullResponseStartFrame())
            try:
                async for token in reply_tokens(self._delays, self._counters):
                    await self._push_llm_text(token)
            finally:
                await self.push_frame(f.LLMFullResponseEndFrame())

    class FakeTTS(TTSService):
        def __init__(self, delays: Delays) -> None:
            super().__init__(
                sample_rate=RATE,
                push_start_frame=True,
                push_stop_frames=True,
                settings=TTSSettings(model=None, voice=None, language=None),
            )
            self._delays = delays

        async def run_tts(self, text: str, context_id: str) -> Any:
            await asyncio.sleep(self._delays.tts_first_ms / 1000)
            pcm, step = tts_audio(text, self._delays), FRAME_BYTES * 5
            for i in range(0, len(pcm), step):
                yield f.TTSAudioRawFrame(pcm[i : i + step], RATE, 1, context_id=context_id)

    async def build(setup: Setup, scenario: Scenario, delays: Delays, **kw: Any) -> Any:
        """The caller's input, a runner holding the pipeline, and what this setup changed."""
        notes: list[str] = []
        user = LLMUserAggregatorParams()
        if setup.provider_tricks:
            # Pipecat's Flux recipe: no user_turn_strategies, so the STT's recommendation
            # (EagerUserTurnStrategies: external start/stop + eager speculation) is applied.
            notes.append(
                f"Flux-like STT: proposed start after {FLUX_START_MS:.0f} ms speech, eager end "
                f"{delays.flux_eager_ms:.0f} ms + EndOfTurn {delays.flux_end_ms:.0f} ms after "
                "speech; EagerUserTurnStrategies recommended by the STT; no local VAD"
            )
        else:
            vad = VADParams() if setup.end_ms is None else VADParams(stop_secs=setup.end_ms / 1000)
            turn = OracleSmartTurn(kw["done"])
            turn.warm()
            user = LLMUserAggregatorParams(
                vad_analyzer=EnergyVAD(params=vad),
                user_turn_strategies=UserTurnStrategies(
                    start=[VADStart(), TranscriptionUserTurnStartStrategy()],
                    stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn)],
                ),
            )
            notes.append(
                f"VAD start {vad.start_secs}s stop {vad.stop_secs}s; default stop strategy: "
                "smart-turn v3 (real model run, verdict = caller finished); Deepgram-like STT"
            )
        stt = FakeSTT(scenario, delays, flux=setup.provider_tricks)
        pair = LLMContextAggregatorPair(LLMContext(), user_params=user)
        audio_in = CallerInput(TransportParams(audio_in_enabled=True, audio_in_sample_rate=RATE))
        out = TransportParams(
            audio_out_enabled=True, audio_out_sample_rate=RATE, audio_out_end_silence_secs=0
        )
        stages = [audio_in, stt, pair.user(), FakeLLM(delays, kw["counters"]), FakeTTS(delays)]
        stages += [Speaker(out, kw["playout"]), pair.assistant()]
        task = PipelineWorker(
            Pipeline(stages),
            params=PipelineParams(audio_in_sample_rate=RATE, audio_out_sample_rate=RATE),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(task)
        return audio_in, runner, task, notes

    return build


async def run_trial(setup: Setup, scenario: Scenario, delays: Delays) -> Trial:
    trial = Trial(framework="pipecat", setup=setup.name, scenario=scenario.name)
    counters, playout, marks = Counters(), Playout(), {}
    done, barge = asyncio.Event(), asyncio.Event()
    last_speech = max(i for i, (_, speech) in enumerate(caller_frames(scenario)) if speech)
    sent = 0
    running: asyncio.Task | None = None
    barger: asyncio.Task | None = None
    try:
        audio_in, runner, task, trial.notes = await _pipecat()(
            setup, scenario, delays, done=done, counters=counters, playout=playout
        )
        running = asyncio.create_task(runner.run())
        await asyncio.wait_for(audio_in.ready.wait(), 10)
        await asyncio.sleep(0.1)  # let the StartFrame's metadata (turn strategies) settle

        async def push(pcm: bytes) -> None:
            nonlocal sent
            await audio_in.push(pcm)
            if sent == last_speech:
                done.set()
            sent += 1

        async def cut_in() -> None:
            await playout.on_first.wait()
            await asyncio.sleep((scenario.barge_after_ms or 0) / 1000)
            barge.set()

        if scenario.barge_after_ms is not None:
            barger = asyncio.create_task(cut_in())
        cpu = time.process_time()
        await send_caller(scenario, push, marks, barge if barger else None)
        trial.cpu_us_per_frame = round((time.process_time() - cpu) * 1e6 / max(1, sent), 1)
        await task.cancel()
        await asyncio.wait_for(running, 10)
    except Exception as exc:  # a broken trial is a result, not a crash
        trial.error = f"{type(exc).__name__}: {exc}"
    finally:
        for left in (barger, running):
            if left and not left.done():
                left.cancel()
    score(trial, marks, playout)
    trial.llm_calls, trial.tokens = counters.llm_calls, counters.tokens
    return trial


async def _main() -> None:
    delays = Delays()
    for setup in SETUPS:
        for scenario in SCENARIOS:
            print(await run_trial(setup, scenario, delays), flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
