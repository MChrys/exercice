import asyncio
import httpx
import random
import re
import time
import aiofiles
from tqdm.asyncio import tqdm_asyncio
from transformers import LlamaTokenizerFast
from langchain.prompts import PromptTemplate
from openai import AsyncOpenAI
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import io
import re
import streamlit as st
import pathlib
import os
import spacy
import epitran
import json
import streamlit_authenticator as stauth
import yaml
import whisperx
import torch
import gc
from yaml.loader import SafeLoader
from collections import defaultdict
from similarity.jarowinkler import JaroWinkler


class AsyncResultDict(dict):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, asyncio.Task):
            if value.done():
                return value.result()
            else:
                return value
        return value

    def __str__(self):
        return self._format_dict()

    def __repr__(self):
        return self._format_dict()

    def _format_dict(self):
        formatted = {}
        for key, value in self.items():
            if isinstance(value, asyncio.Task):
                if value.done():
                    formatted[key] = value.result()
                else:
                    formatted[key] = f"<Task pending>"
            else:
                formatted[key] = value
        return repr(formatted)

class ParallelLLMInference:
    def __init__(self, 
                 base_url,exaion_model_name, 
                 exaion_api_key, 
                 hf_model_name,  
                 max_tokens, 
                 max_concurrent_requests, 
                 system_prompt_path, 
                 system_placeholder, 
                 user_prompt_path):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=exaion_api_key,
            timeout=60
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(hf_model_name)
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.exaion_model_name = exaion_model_name 
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        #self.normalisation_prompt_path = normalisation_prompt_path

        self.system_placeholder = system_placeholder


    def chunk_speech(self,text):

        chunks = [[]] 
        prev = 0
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_tokens * 0.9:
            return [text]
        end_of_sentence_tokens = set([self.tokenizer.encode('test.', add_special_tokens=False)[-1],
                                     self.tokenizer.encode('test!', add_special_tokens=False)[-1],
                                     self.tokenizer.encode('test?', add_special_tokens=False)[-1]
                                     ])
        for i in range(len(tokens)):
            cur_sentence_len = i - prev + 1
            if tokens[i] in end_of_sentence_tokens or cur_sentence_len == self.max_tokens *0.9:
                if len(chunks[-1]) and len(chunks[-1]) + cur_sentence_len > self.max_tokens*0.9:
                    chunks.append([])
                chunks[-1].extend(tokens[prev:i+1])
                prev = i + 1

        if prev < len(tokens):
            if len(chunks[-1]) and len(chunks[-1]) + i - prev + 1 > self.max_tokens*0.9:
                chunks.append([])
            chunks[-1].extend(tokens[prev:i+1])

        return [self.tokenizer.decode(chunk, add_special_tokens=False)for chunk in chunks]
    
    async def get_prompt_from_file(self, file_path):
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            return await file.read()

    async def infer_llm(self, text, system_prompt_path, system_placeholder, user_prompt_path):
        async with self.semaphore:
            system_prompt_template = await self.get_prompt_from_file(system_prompt_path)
            #normalisation_template = await self.get_prompt_from_file(normalisation_path)
            user_prompt = await self.get_prompt_from_file(user_prompt_path)

            system_prompt = PromptTemplate.from_template(system_prompt_template).format(system_value=system_placeholder)
            normalisation = PromptTemplate.from_template(user_prompt).format(text=text)

            retries = 5
            backoff_factor = 0.5

            for attempt in range(retries):
                wait_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                try:
                    response = await self.client.chat.completions.create(
                        model=self.exaion_model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        stream=False,
                        max_tokens=self.max_tokens
                    )
                    x = response.choices[0].message.content.strip()
                    return response.choices[0].message.content.strip()
                except (httpx.HTTPStatusError, asyncio.TimeoutError, httpx.RemoteProtocolError) as e:
                    print(f"Error: {e}, retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    print(f"Unexpected error: {e}, retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
            raise Exception("Max retries exceeded for API call")
    
    async def queue_api_calls(self, chunks, pbar):
        tasks = [
            self.infer_llm(chunk, self.system_prompt_path, self.system_placeholder, self.user_prompt_path)
            for chunk in chunks
        ]
        responses = await asyncio.gather(*tasks)
        pbar.update(len(responses))
        r= responses
        return " ".join(responses).strip()
    

    async def parallel_inference(self, segments):
        tasks = []
        processed_segments = []
        chunks_process = [(len(chunks := self.chunk_speech(segment["text"])), chunks,segment["speaker"]) for segment in segments]
        total_chunks = sum(length for length, _, _ in chunks_process)

        with tqdm_asyncio(total=total_chunks, desc="Processing") as pbar:
            for _ ,chunks,speaker in chunks_process:
                
                future = asyncio.ensure_future(self.queue_api_calls(chunks, pbar))
                processed_segment = AsyncResultDict({
                    "speaker": speaker,
                    "text": future
                })
                processed_segments.append(processed_segment)
                tasks.append(future)

        await asyncio.gather(*tasks)

        x = processed_segments
        return processed_segments


    async def LLM_inference(self, segments):
        start = time.time()
        results = await self.parallel_inference(segments)
        # results = [{k: v.strip() if k == "text" else v for k, v in r.items()} for r in results]
        end = time.time()
        print(f"Processing time: {end - start} seconds")
        return results

# Example usage
if __name__ == "__main__":
    initialize(config_path="config")
    cfg = compose(config_name="local")

    api_key = cfg["llm_api"]["api_key"]
    llm_model_name = cfg["llm_api"]["llm_model_name"]
    base_url = cfg["llm_api"]["api_url"]

    senators_file_path = cfg["senators_file_path"]

    hf_model_name = cfg["llm_api"]["hf_model_name"]
    max_tokens = cfg["max_tokens"]
    max_concurrent_requests = cfg["max_concurrent_requests"]

    exaion_model_name = llm_model_name
    exaion_api_key = api_key
   
    #hf_token = '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'

    system_prompt_path = "services/system_prompt.txt"
    user_prompt_path = "services/prompt_normalisation_v0.txt"
    #normalisation_prompt_path = "services/prompt_normalisation_v0.txt"
    system_placeholder = "rédaction de compte rendu"

    inference = ParallelLLMInference(base_url, 
                                     exaion_model_name, 
                                     exaion_api_key, hf_model_name, 
                                     max_tokens, max_concurrent_requests, 
                                     system_prompt_path, 
                                     system_placeholder, 
                                     user_prompt_path,   
                                     )
    
    # Example segments to infer
    segments = [
        {"speaker": "Speaker 1", "text": "Hello, this is a test."},
        {"speaker": "Speaker 2", "text": "Hi, this is another test."}
    ]
    
    results = inference.LLM_inference(segments)
    print("results : ")
    print(results)
