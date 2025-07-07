import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def call_qwen_api(prompt, system_prompt="You are a helpful assistant."):
    """
    Calls the Qwen model via Nebius AI API.

    To use this function, you must set the NEBIUS_API_KEY environment variable.
    For example, in your terminal:
    export NEBIUS_API_KEY='your_nebius_api_key'
    """
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY")
    )

    if not client.api_key:
        raise ValueError("NEBIUS_API_KEY environment variable not set.")

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-7B",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return response.choices[0].message.content
