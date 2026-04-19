from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5-nano",
    input="Explain RAG in one line",
    max_output_tokens=200,
    reasoning={"effort": "low"}
)

print(response.output_text)
print(response.usage)