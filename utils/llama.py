import backoff  # for exponential backoff
import openai
import os
import asyncio
from typing import Any
from .data_reading import read_openapi
from llama_cpp import Llama
import tiktoken
from typing import Optional, Union
"""
Candidate Llama2 models
/hdd2/lh/project/llms/l2/llama-2-13b-chat.Q5_K_M.gguf  
/hdd2/lh/project/llms/l2/llama-2-7b-chat.Q5_K_M.gguf
"""
LLAMA_MODELS = {"13B":"/hdd2/lh/project/llms/l2/llama-2-13b-chat.Q5_K_M.gguf",
        "7B":"/hdd2/lh/project/llms/l2/llama-2-7b-chat.Q5_K_M.gguf"
         }


def llama_v2_prompt(messages: list[dict]):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


async def dispatch_openai_chat_requests(
        messages_list: list[list[dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_words: list[str]
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def dispatch_openai_prompt_requests(
        messages_list: list[list[dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_words: list[str],
        logprobs:Optional[Union[int, None]],
        echo: bool
) -> list[str]:
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop_words,
            logprobs = logprobs,
            echo= echo
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


class Llama2:
    def __init__(self, model_name='7B', n_ctx=512, n_gpu_layers=-1, logits_all=False,
                 stop_words=[], max_new_tokens=128, logprobs=None, echo=False):
        self.model_path = LLAMA_MODELS[model_name]
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        self.logprobs = logprobs
        self.echo = echo
        self.llm = Llama(model_path=self.model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, logits_all=logits_all)

    # used for chat-gpt and gpt-4 gpt-3.5-turbo-0301
    def chat_generate(self, input_string, temperature=0.0):
        response = self.llm(prompt=input_string, max_tokens=self.max_new_tokens, temperature=temperature)
        response = chat_completions_with_backoff(
            model=self.model_name,
            messages=[
                {"role": "user", "content": input_string}
            ],
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=self.stop_words
        )
        # generated_text = response['choices'][0]['message']['content'].strip()
        return response

    # used for text-davinci-003
    def prompt_generate(self, input_string, temperature=0.0):
        response = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=self.stop_words,
            logprobs=self.logprobs,
            echo=self.echo
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text

    def batch_chat_generate(self, messages_list, temperature=0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return predictions
        # return [x['choices'][0]['message']['content'].strip() for x in predictions]

    def batch_prompt_generate(self, prompt_list, temperature=0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words,
                logprobs=self.logprobs, echo=self.echo
            )
        )
        return predictions

    def batch_generate(self, messages_list, temperature=0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ['gpt-3.5-turbo']:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate(self, input_string, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string, suffix, temperature=0.0):
        response = completions_with_backoff(
            model=self.model_name,
            prompt=input_string,
            suffix=suffix,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text

    def token_counting(self, input):
        if self.model_name in ['text-davinci-002', 'text-davinci-003']:
            encoding = tiktoken.get_encoding("p50k_base")
        elif self.model_name in ['gpt-3.5-turbo', 'gpt-4', 'text-embedding-ada-002']:
            encoding = tiktoken.get_encoding("cl100k_base")
        elif self.model_name in ['davinci']:
            encoding = tiktoken.get_encoding("gpt2")
        else:
            "wrong model for token counting"
            exit()
        num_tokens = encoding.encode(input)
        return num_tokens

if __name__ == "__main__":
    api = read_openapi()
    openai.api_key = api
    openai.Completion.create(
        model="text-davinci-003",
        prompt="what is the conce=70",
        temperature=0,
        logprobs=1,
        echo=True
    )
    print("sdf")