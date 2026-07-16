import streamlit as st
import validators

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Streamlit Configuration
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


# Sidebar
with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )


# URL Input
url = st.text_input(
    "Enter YouTube or Website URL"
)


# Prompt
prompt_template = """
Provide a clear summary of the following content in around 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# Button
if st.button("Summarize the Content"):

    if not groq_api_key:
        st.error("Please enter your Groq API Key")
        st.stop()

    if not url:
        st.error("Please enter a URL")
        st.stop()

    if not validators.url(url):
        st.error("Please enter a valid URL")
        st.stop()


    try:

        with st.spinner("Processing..."):

            # Groq LLM
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_api_key,
                temperature=0
            )


            # Load content

            if "youtube.com" in url or "youtu.be" in url:

                st.info("Loading YouTube transcript...")

                loader = YoutubeLoader.from_youtube_url(
                    url,
                    add_video_info=False
                )

            else:

                st.info("Loading webpage...")

                loader = UnstructuredURLLoader(
                    urls=[url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )


            docs = loader.load()


            if not docs:
                st.error("Could not extract content from this URL")
                st.stop()


            # Split large documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=200
            )

            docs = text_splitter.split_documents(docs)


            st.success(
                f"Loaded {len(docs)} document chunks"
            )


            # Summarization chain
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=prompt,
                combine_prompt=prompt
            )


            result = chain.invoke(
                {
                    "input_documents": docs
                }
            )


            st.subheader("Summary")
            st.write(result["output_text"])


    except Exception as e:

        st.error("Something went wrong:")
        st.exception(e)