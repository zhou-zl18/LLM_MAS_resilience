temperature = 0.0

aliyun_key = "set your own key"
qwen32b_config = {"model": 'qwen2.5-32b-instruct', 
            "api_key": aliyun_key, 
            "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
            "temperature": temperature,
            "cache_seed": None,
            "price" : [0.0, 0.0]}

openai_key = "set your own key"
openai_base_url = "set your own base url"
gpt35_config = {"model": "gpt-35-turbo",
            "api_type": "azure",
            "api_key": openai_key,
            "temperature": temperature,
            "cache_seed": None,
            "base_url": openai_base_url,
            "api_version": "2025-01-01-preview"}
gpt4omini_config = {"model": "gpt-4o-mini",
            "api_type": "azure",
            "api_key": openai_key,
            "temperature": temperature,
            "cache_seed": None,
            "base_url": openai_base_url,
            "api_version": "2025-01-01-preview"}


from openai import AzureOpenAI
from openai import AsyncAzureOpenAI
import asyncio
async def get_completion(prompt, model='gpt-4o-mini', temperature=0):
    client = AsyncAzureOpenAI(
        api_key = openai_key,
        api_version = "2025-01-01-preview",
        azure_endpoint = openai_base_url,
    )
    messages = [{"role": "user", "content": prompt}]
    import time
    for attempt in range(10):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            break
        except Exception as e:
            if attempt < 9:
                print(f"Attempt {attempt+1} failed with error: {e}. Retrying in 30 seconds...")
                await asyncio.sleep(30)
            else:
                print(f"All 10 attempts failed. Last error: {e}")
                raise
    return response.choices[0].message.content