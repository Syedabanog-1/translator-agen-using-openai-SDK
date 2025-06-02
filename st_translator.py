import streamlit as st
import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, AsyncOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file")

# Set up external Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Set up the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Configure the runner
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Translator Agent instructions
translator_agent = Agent(
    name='Translator Agent',
    instructions=(
        "You are a translator agent. Translate the given input text "
        "to the specified target language. Respond ONLY with the translated text, "
        "no explanations or extra information."
    )
)

# Async wrapper for translation
async def _run_translation(prompt: str) -> str:
    response = await Runner.run(
        translator_agent,
        input=prompt,
        run_config=config
    )
    return response.final_output.strip()

# Sync translation function for Streamlit
def translate_text(text: str, target_language: str) -> str:
    prompt = f"Translate this text to {target_language}:\n\n{text}"
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(_run_translation(prompt))

# Streamlit app
def main():
    st.title("🌍 Translator Agent")

    input_text = st.text_area("Enter text to translate", height=150)

    languages = [
        "English", "Spanish", "French", "German", "Chinese", "Japanese", "Russian",
        "Arabic", "Hindi", "Portuguese", "Bengali", "Korean", "Italian", "Dutch", "Turkish",
        "Swedish", "Vietnamese", "Polish", "Thai", "Persian", "Indonesian"
    ]
    target_language = st.selectbox("Select target language", options=languages)

    if st.button("Translate"):
        if not input_text.strip():
            st.error("Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                try:
                    translated = translate_text(input_text, target_language)
                    st.success("Translation complete!")
                    st.text_area("Translated Text", value=translated, height=150)
                except Exception as e:
                    st.error(f"Translation failed: {e}")

if __name__ == "__main__":
    main()
