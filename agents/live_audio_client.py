"""Streaming helper for Gemini Live API audio workflows."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Sequence

import google.genai.types as genai_types
from google import genai


DEFAULT_AUDIO_MIME = "audio/wav"
DEFAULT_STREAM_CHUNK_SIZE = 32_000  # ~0.5s of 16 kHz mono PCM


class GeminiLiveAudioClient:
    """
    Wraps Gemini's Live API so the tutor agent can synthesize speech, stream
    learner audio, and receive real-time analysis directly from the model.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        voice: str = "Studio",
        chunk_size: int = DEFAULT_STREAM_CHUNK_SIZE,
    ) -> None:
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable must be set for Live API usage."
            )

        self._client = genai.Client(api_key=resolved_key)
        self._model_name = model
        self._voice = voice
        self._chunk_size = chunk_size

    @asynccontextmanager
    async def conversation_session(
        self,
        *,
        instructions: Optional[str] = None,
        response_modalities: Optional[Sequence[str]] = None,
        audio_voice: Optional[str] = None,
        spoken_language: Optional[str] = None,
    ) -> AsyncIterator["LiveConversationSession"]:
        """
        Open a bidirectional streaming session for continuous conversation.

        Example:

            async with client.conversation_session(
                spoken_language="es"
            ) as convo:
                await convo.send_system_instructions("Be a helpful tutor.")
                await convo.send_text("Hola, ¿cómo estás?")
                async for message in convo.receive():
                    ...
        """

        modalities = list(response_modalities or ["AUDIO", "TEXT"])
        config: Dict[str, object] = {"response_modalities": modalities}

        # Only include audio config if caller expects audio back.
        if "AUDIO" in modalities:
            config["generation_config"] = {
                "audio_config": {
                    "voice": audio_voice or self._voice,
                    "format": "wav",
                    "spoken_language": spoken_language or "en",
                }
            }

        async with self._connect(config=config) as session:
            convo = LiveConversationSession(
                session=session,
                chunk_size=self._chunk_size,
            )

            if instructions:
                await convo.send_system_instructions(instructions)

            yield convo

    async def synthesize_phrase_audio(self, phrase: str, language_code: str) -> bytes:
        """
        Stream TTS audio straight from the Live API and return concatenated bytes.
        """
        config = {
            "response_modalities": ["AUDIO"],
            "generation_config": {
                "audio_config": {
                    "voice": self._voice,
                    "format": "wav",
                    "spoken_language": language_code,
                }
            },
        }

        async with self._connect(config=config) as session:
            await session.send_client_content(
                turns=[
                    genai_types.Content(
                        role="system",
                        parts=[
                            genai_types.Part(
                                text=(
                                    "You stream pronunciation-perfect speech audio for "
                                    "language learners. Speak phrases exactly as given "
                                    "without explanations."
                                )
                            )
                        ],
                    ),
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                text=(
                                    f"Language code: {language_code}\n"
                                    f"Speak this phrase: {phrase}"
                                )
                            )
                        ],
                    ),
                ]
            )
            audio = await self._collect_audio(session)

        if not audio:
            raise RuntimeError("Live API returned no audio for phrase synthesis.")
        return audio

    async def analyze_pronunciation(
        self,
        audio_bytes: bytes,
        target_text: str,
        language_code: str,
    ) -> Dict[str, object]:
        """
        Stream learner speech to Gemini and return structured pronunciation feedback.
        """
        config = {
            "response_modalities": ["TEXT"],
            "generation_config": {"temperature": 0.2},
        }
        instructions = (
            "You are a pronunciation coach. Compare the learner audio to the provided "
            "target phrase. Respond with compact JSON using this schema:\n"
            "{\n"
            '  "accuracy": float 0..1,\n'
            '  "is_correct": bool,\n'
            '  "feedback": str,\n'
            '  "problematic_words": [str],\n'
            '  "suggestions": [str],\n'
            '  "transcription": str\n'
            "}\n"
            "Do not include any additional commentary."
        )

        async with self._connect(config=config) as session:
            await session.send_client_content(
                turns=[
                    genai_types.Content(
                        role="system",
                        parts=[genai_types.Part(text=instructions)],
                    ),
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                text=(
                                    f"Language code: {language_code}\n"
                                    f"Target phrase: {target_text}\n"
                                    "Begin evaluating once the audio stream finishes."
                                )
                            )
                        ],
                    ),
                ],
                turn_complete=False,
            )

            await self._stream_audio_chunks(session, audio_bytes)
            await session.send_client_content(turns=None, turn_complete=True)
            response_text = await self._collect_text(session)

        return self._parse_analysis_json(response_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def _connect(self, config: Dict[str, object]):
        session_cm = self._client.aio.live.connect(model=self._model_name, config=config)
        async with session_cm as session:
            yield session

    async def _collect_audio(self, session) -> bytes:
        chunks: List[bytes] = []
        async for message in session.receive():
            data = getattr(message, "data", None)
            if data:
                chunks.append(data)
        return b"".join(chunks)

    async def _collect_text(self, session) -> str:
        fragments: List[str] = []
        async for message in session.receive():
            if message.text:
                fragments.append(message.text)
        combined = "".join(fragments).strip()
        if not combined:
            raise RuntimeError("Live API did not return textual feedback.")
        return combined

    async def _stream_audio_chunks(self, session, audio_bytes: bytes) -> None:
        for chunk in self._chunk_bytes(audio_bytes):
            await session.send_realtime_input(
                audio=genai_types.Blob(data=chunk, mime_type=DEFAULT_AUDIO_MIME)
            )
        await session.send_realtime_input(audio_stream_end=True)

    def _chunk_bytes(self, payload: bytes) -> List[bytes]:
        if not payload:
            return []
        return [
            payload[i : i + self._chunk_size]
            for i in range(0, len(payload), self._chunk_size)
        ]

    @staticmethod
    def _parse_analysis_json(raw_text: str) -> Dict[str, object]:
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        return {
            "accuracy": 0.0,
            "is_correct": False,
            "feedback": raw_text or "Unable to analyze pronunciation.",
            "problematic_words": [],
            "suggestions": [],
            "transcription": "",
        }


@dataclass
class LiveConversationSession:
    """
    Thin wrapper around the Live API session object that exposes ergonomic helpers
    for continuous microphone-style conversations.
    """

    session: object
    chunk_size: int

    async def send_system_instructions(self, text: str) -> None:
        await self.send_text(text, role="system", turn_complete=False)

    async def send_text(
        self,
        text: str,
        *,
        role: str = "user",
        turn_complete: bool = False,
    ) -> None:
        await self.session.send_client_content(
            turns=[
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=text)],
                )
            ],
            turn_complete=turn_complete,
        )

    async def stream_audio(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str = DEFAULT_AUDIO_MIME,
        end_stream: bool = True,
    ) -> None:
        for chunk in self._chunk_bytes(audio_bytes):
            await self.session.send_realtime_input(
                audio=genai_types.Blob(data=chunk, mime_type=mime_type)
            )
        if end_stream:
            await self.session.send_realtime_input(audio_stream_end=True)

    async def end_turn(self) -> None:
        await self.session.send_client_content(turns=None, turn_complete=True)

    def receive(self) -> AsyncIterator[genai_types.LiveServerMessage]:
        return self.session.receive()

    def _chunk_bytes(self, payload: bytes) -> List[bytes]:
        if not payload:
            return []
        return [
            payload[i : i + self.chunk_size]
            for i in range(0, len(payload), self.chunk_size)
        ]


