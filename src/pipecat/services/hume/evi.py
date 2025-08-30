import asyncio

from hume import AsyncHumeClient
from loguru import logger

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (AudioRawFrame, EndFrame, ErrorFrame, Frame, HumeVoiceFrame, StartFrame, TextFrame)

class HumeEVI(FrameProcessor):
    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self._client = None
        self._task = None
        self._socket = None
        self._audio_out_sample_rate = None  # Store sample rate from StartFrame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._client = AsyncHumeClient(self._api_key)
            self._audio_out_sample_rate = getattr(frame, "audio_out_sample_rate", 16000)
            if not self._task:
                self._task = asyncio.create_task(self._run_hume())
        elif isinstance(frame, EndFrame):
            await self._stop()

        if isinstance(frame, AudioRawFrame):
            await self._send_audio(frame)
        elif isinstance(frame, TextFrame):
            await self._send_text(frame)

    async def _stop(self):
        if self._task:
            self._task.cancel()
            self._task = None
        await super().cleanup()

    async def _send_audio(self, frame: AudioRawFrame):
        if not self._socket:
            return

        try:
            await self._socket.send_audio(frame.audio)
        except Exception as e:
            logger.error(f"Error sending audio to Hume: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))

    async def _send_text(self, frame: TextFrame):
        if not self._socket:
            return

        try:
            logger.info(f"Sending text to Hume: {frame.text}")
            await self._socket.send_text(frame.text)
        except Exception as e:
            logger.error(f"Error sending text to Hume: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))

    async def _run_hume(self):
        try:
            async with self._client.empathic_voice.chat() as socket:
                self._socket = socket
                async for message in socket:
                    if message.type == "audio_output":
                        audio = message.data
                        await self.push_frame(
                            AudioRawFrame(
                                audio=audio,
                                sample_rate=self._audio_out_sample_rate or 16000,
                                num_channels=1,
                            )
                        )
                    else:
                        await self.push_frame(HumeVoiceFrame(data=str(message)))
        except Exception as e:
            logger.error(f"Hume connection error: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))