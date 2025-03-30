import os
from dotenv import load_dotenv
from activevision_agent.states import ActiveVisionOverallState
from groq import Groq

load_dotenv()
MODEL ="llama-3.3-70b-specdec"

def describe_query(state: ActiveVisionOverallState) -> ActiveVisionOverallState:
    client = Groq(api_key=os.getenv('GROQ'))
    prompt = f"Provide a detailed description for the following query: {state.query}"
    response = client.chat.completions.create(
        model= MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=150
    )
    description = ""
    for chunk in response:
        description += (lambda: chunk.choices[0].delta.content if chunk.choices[0].delta.content else "")()
    state.query = description

    return state
