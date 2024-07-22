from openai import OpenAI

client = OpenAI(
    base_url="https://chat.ai.linagora.exaion.com/v1/",
    api_key = 'sk-rTWvMe7RJXofttd4p4tkkQ'
)
chat_completion = client.chat.completions.create(
    model="2-meta-llama-3-8b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why is open-source software important?"},
    ],
    stream=True,
    max_tokens=500
)

# iterate and print stream
for message in chat_completion:
    print(message.choices[0].delta.content, end="")
