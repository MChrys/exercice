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

class ParallelLLMInference:
    def __init__(self, exaion_model_name, exaion_api_key, hf_model_name, hf_token, max_tokens, max_concurrent_requests, system_prompt_path, system_placeholder, user_prompt_path):
        self.client = AsyncOpenAI(
            base_url="https://chat.ai.linagora.exaion.com/v1/",
            api_key=exaion_api_key,
            timeout=60
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(hf_model_name, token=hf_token, cache_dir='../../.cache/huggingface')
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.exaion_model_name = exaion_model_name 
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        self.system_placeholder = system_placeholder

    def chunk_speech(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_tokens * 0.9:
            return [text]
        
        sentence_endings = r"(?<=[.!?]) +"
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            if current_chunk_tokens + len(sentence_tokens) <= self.max_tokens:
                current_chunk += sentence + " "
                current_chunk_tokens += len(sentence_tokens)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                current_chunk_tokens = len(sentence_tokens)

        if current_chunk:
            chunks.append(current_chunk.strip())

        final_chunks = []
        for chunk in chunks:
            chunk_tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
            if len(chunk_tokens) > self.max_tokens * 0.9:
                words = chunk.split()
                current_chunk = ""
                current_chunk_tokens = 0
                for word in words:
                    word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
                    if current_chunk_tokens + len(word_tokens) <= self.max_tokens:
                        current_chunk += word + " "
                        current_chunk_tokens += len(word_tokens)
                    else:
                        final_chunks.append(current_chunk.strip())
                        current_chunk = word + " "
                        current_chunk_tokens = len(word_tokens)
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks
    
    async def get_prompt_from_file(self, file_path):
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            return await file.read()

    async def infer_llm(self, text, system_prompt_path, system_placeholder, user_prompt_path):
        async with self.semaphore:
            system_prompt_template = await self.get_prompt_from_file(system_prompt_path)
            user_prompt_template = await self.get_prompt_from_file(user_prompt_path)

            system_prompt = PromptTemplate.from_template(system_prompt_template).format(system_value=system_placeholder)
            user_prompt = PromptTemplate.from_template(user_prompt_template).format(text=text)

            retries = 5
            backoff_factor = 0.5

            for attempt in range(retries):
                wait_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                try:
                    response = await self.client.chat.completions.create(
                        model=self.exaion_model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "user", "content": text},
                        ],
                        stream=False,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content.strip()
                except (httpx.HTTPStatusError, asyncio.TimeoutError, httpx.RemoteProtocolError) as e:
                    print(f"Error: {e}, retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    print(f"Unexpected error: {e}, retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
            raise Exception("Max retries exceeded for API call")

    async def queue_api_calls(self, index, speaker, chunks, pbar):
        tasks = [
            self.infer_llm(chunk, self.system_prompt_path, self.system_placeholder, self.user_prompt_path)
            for chunk in chunks
        ]
        responses = await asyncio.gather(*tasks)
        for _ in responses:
            pbar.update(1)
        return [(index, speaker, i, response) for i, response in enumerate(responses)]

    async def parallel_inference(self, segments):
        tasks = []
        total_chunks = sum(len(self.chunk_speech(segment["text"])) for segment in segments)

        with tqdm_asyncio(total=total_chunks, desc="Processing") as pbar:
            for i, segment in enumerate(segments):
                speaker = segment["speaker"]
                text = segment["text"]
                chunks = self.chunk_speech(text)
                tasks.append(self.queue_api_calls(i, speaker, chunks, pbar))
            results = await asyncio.gather(*tasks)
        
        # Flatten and sort the results by index and chunk index to maintain the original order
        processed_segments = {}
        for result in results:
            for idx, speaker, chunk_idx, response in sorted(result, key=lambda x: (x[0], x[2])):
                if idx not in processed_segments:
                    processed_segments[idx] = {"speaker": speaker, "text": ""}
                processed_segments[idx]["text"] += " " + response
        
        # Convert to list and strip extra spaces
        final_segments = [
            {"speaker": data["speaker"], "text": data["text"].strip()}
        for _, data in sorted(processed_segments.items())
        ]
        
        return final_segments

    def LLM_inference(self, segments):
        start = time.time()
        results = asyncio.run(self.parallel_inference(segments))
        # results = [{k: v.strip() if k == "text" else v for k, v in r.items()} for r in results]
        end = time.time()
        print(f"Processing time: {end - start} seconds")
        return results

# Example usage
if __name__ == "__main__":
    exaion_model_name = 'meta-llama-3-8b-instruct'
    exaion_api_key = "sk-7Gqg14u-mGlX-egix20lgg"
    hf_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    hf_token = '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'
    max_tokens = 1024
    max_concurrent_requests = 5

    system_prompt_path = "services/system_prompt.txt"
    user_prompt_path = "services/user_prompt.txt"
    system_placeholder = "rédaction de compte rendu"

    inference = ParallelLLMInference(exaion_model_name, exaion_api_key, hf_model_name, hf_token, max_tokens, max_concurrent_requests, system_prompt_path, system_placeholder, user_prompt_path)
    
    # Example segments to infer
    segments = [
        {"speaker": "Speaker 1", "text": "Hello, this is a test."},
        {"speaker": "Speaker 2", "text": "Hi, this is another test."}
    ]
    
    results = inference.LLM_inference(segments)
    print(results)
