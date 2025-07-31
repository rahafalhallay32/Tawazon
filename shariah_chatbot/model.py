from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key="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"
)

def generate_answer(question: str) -> str:
    """
    Sends a question to GPT-4o with strict instruction to only answer based on AAOIFI standards.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                "You are a professional assistant specialized in Islamic finance, particularly in AAOIFI Shariah standards "
                "(Accounting and Auditing Organization for Islamic Financial Institutions). "
                "You help users understand and apply these standards clearly and accurately. "
                "Introduce yourself politely only in the first user interaction, then directly answer future questions without repeating introductions. "
                "Use a helpful and respectful tone. If a question is completely unrelated to Islamic finance, politely inform the user."
             )
            },
            {
                "role": "user",
                "content": f"Question: {question}"
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            timeout=60
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        raise

