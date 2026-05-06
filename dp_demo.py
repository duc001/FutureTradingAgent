from openai import OpenAI

client = OpenAI(
    api_key="sk-205ee372c5df491f8050324eb697e504",
    base_url="https://api.deepseek.com")


response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好"},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

print(response.choices[0].message.content)
print("--------")
print(response.choices[0].message.reasoning_content)

