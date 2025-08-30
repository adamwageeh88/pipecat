
from typing import AsyncGenerator

from litellm import acompletion
from loguru import logger

from pipecat.frames.frames import LLMContextFrame, TextFrame, Frame
from pipecat.services.llm_service import LLMService
from pipecat.processors.frame_processor import FrameDirection

class LiteLLMService(LLMService):
    def __init__(self, model: str = "ollama/llama3"): # Default to a local Ollama model
        super().__init__()
        self._model = model

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            logger.debug(f"LiteLLMService: Received LLMContextFrame: {frame.context.get_messages()}")
            response_content = ""
            try:
                response = await acompletion(
                    model=self._model,
                    messages=frame.context.get_messages(), # Use messages from the context frame
                    stream=True,
                    api_base="http://localhost:11434" # Explicitly set API base for Ollama
                )
                async for chunk in response:
                    logger.debug(f"Received chunk from LiteLLM: {chunk}")
                    content = chunk.choices[0].delta.content
                    if content:
                        response_content += content
                        logger.debug(f"LiteLLMService: Pushing TextFrame: {content}")
                        await self.push_frame(TextFrame(content))
            except Exception as e:
                logger.error(f"Error calling LiteLLM: {e}")
                response_content = f"Error: {e}"
                await self.push_frame(TextFrame(response_content))

            # Push an LLMContextFrame back to the assistant aggregator
            frame.context.add_message({"role": "assistant", "content": response_content})
            logger.debug(f"LiteLLMService: Pushing LLMContextFrame to assistant: {frame.context.get_messages()}")
            await self.push_frame(LLMContextFrame(frame.context))
        else:
            await self.push_frame(frame)
