import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from langchain_core.tools import tool


from langchain.llms import Ollama


load_dotenv()

llm = Ollama(
    model="gemma2",
    temperature=0.2,
)


prompt = PromptTemplate.from_template(
    """
You are an expert startup analyst.

Extract the following tags from the startup idea below:
1. Domain (e.g., AI, Fintech, SaaS, HealthTech, etc.)
2. Stage (Pre-Seed, Seed, Series A, Series B, Growth)
3. Region (e.g., India, USA, Europe, Global)

Startup Idea:
\"\"\"{idea}\"\"\"

Respond only with JSON:
{{
  "domain": [...],
  "stage": "...",
  "region": "..."
}}
"""
)


parser = JsonOutputParser()


tag_chain: Runnable = prompt | llm | parser


@tool
def extract_tags_tool(idea: str) -> dict:
    """
    Extract startup domain, stage, and region tags from a startup idea text.

    Args:
        idea (str): The startup idea text.

    Returns:
        dict: Parsed JSON output with keys "domain", "stage", and "region".
    """
    return tag_chain.invoke({"idea": idea})


if __name__ == "__main__":
    test_idea = "A SaaS platform that uses AI to automate customer support for startups in the USA at Seed stage."
    tags = extract_tags_tool(test_idea)
    print(tags)
