from typing import AsyncGenerator
import base64

from hume import AsyncHumeClient
from hume.tts import PostedContextWithUtterances, PostedUtterance

from pipecat.frames.frames import AudioRawFrame, Frame, EndFrame
from pipecat.services.tts_service import TTSService
from pipecat.processors.frame_processor import FrameDirection

class HumeTTSService(TTSService):
    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self._client = AsyncHumeClient(api_key=self._api_key)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, EndFrame):
            await self._stop()

    async def _stop(self):
        await super().cleanup()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        utterance = PostedUtterance(text=text)
        # Try passing utterances directly as a keyword argument
        result = await self._client.tts.synthesize_json(utterances=[utterance])
        audio_output_base64 = result.generations[0].audio
        audio_output = base64.b64decode(audio_output_base64)
        yield AudioRawFrame(audio=audio_output, sample_rate=16000, num_channels=1, pts=None)