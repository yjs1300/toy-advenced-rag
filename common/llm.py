from langchain_openai import ChatOpenAI

class LLMFactory:

    @staticmethod
    def create(model = "gpt-4o-mini", temperature = 0.2):
        return ChatOpenAI(model=model, temperature=temperature)