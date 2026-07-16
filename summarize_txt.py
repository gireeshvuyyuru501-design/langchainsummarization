#!/usr/bin/env python3
import os
import sys


from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document


def summarize_file(path): 
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set in environment or .env")
        return 1

    llm = ChatGroq(groq_api_key=api_key, model="Gemma-7b-It")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    docs = [Document(page_content=text)]

    prompt_template = """
Provide a concise summary (around 200-300 words) of the following content:

{text}

Summary:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"]) 

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    summary = chain.run(docs)

    print(summary)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_txt.py path/to/file.txt")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)
    sys.exit(summarize_file(path))
