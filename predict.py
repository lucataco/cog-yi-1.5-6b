# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, ConcatenateIterator
import os
import time
import random
import subprocess
from typing import AsyncIterator, List, Union
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

MODEL_ID = "yi"
WEIGHTS_URL = "https://weights.replicate.delivery/default/01-ai/Yi-1.5-6B/model.tar"

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.2

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class VLLMPipeline:
    """A simplified inference engine that runs inference w/ vLLM"""
    def __init__(self, *args, **kwargs) -> None:
        args = AsyncEngineArgs(*args, **kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = getattr(self.engine.engine.tokenizer, "tokenizer", self.engine.engine.tokenizer)

    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams
    ) -> AsyncIterator[str]:
        results_generator = self.engine.generate(
            prompt, sampling_params, str(random.random())
        )
        async for generated_text in results_generator:
            yield generated_text

    async def __call__(
        self,
        messages: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: float,
        stop_sequences: Union[str, List[str]] = None,
        stop_token_ids: List[int] = None,
        repetition_penalty: float = 1.2,
        incremental_generation: bool = True,
    ) -> str:
        """Given a prompt, runs generation on the language model with vLLM."""
        if top_k is None or top_k == 0:
            top_k = -1

        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_sequences, str) and stop_sequences != "":
            stop = [stop_sequences]
        elif isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            stop = stop_sequences
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        sampling_params = SamplingParams(
            n=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            use_beam_search=False,
            stop=stop,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=True
        )

        generation_length = 0
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        async for request_output in self.generate_stream(text, sampling_params):
            assert len(request_output.outputs) == 1
            generated_text = request_output.outputs[0].text
            if incremental_generation:
                yield generated_text[generation_length:]
            else:
                yield generated_text
            generation_length = len(generated_text)


class Predictor(BasePredictor):
    async def setup(self):
        start = time.time()
        # Download weights
        if not os.path.exists(MODEL_ID):
            download_weights(WEIGHTS_URL, MODEL_ID)
        print(f"downloading weights took {time.time() - start:.3f}s")
        self.llm = VLLMPipeline(
            model=MODEL_ID,
            dtype="auto",
        )

    async def predict(
        self,
        prompt: str = Input(description="Input prompt", default="Tell me a joke"),
        system_prompt: str = Input(description="System prompt", default="You are a friendly Chatbot."),
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=4096,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.", 
            default=DEFAULT_TEMPERATURE, ge=0.1, le=4.0,
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=DEFAULT_TOP_P, ge=0.1, le=1.0,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=DEFAULT_TOP_K,
        ),
    ) -> ConcatenateIterator[str]:
        start = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        generate = self.llm(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        async for text in generate:
            yield text
        print(f"generation took {time.time() - start:.3f}s")
