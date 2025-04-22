from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

import os
from dotenv import load_dotenv
load_dotenv()


def load_bellavista_documents():
    
    embedding_function = AzureOpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        chunk_size=1,
        openai_api_version="2023-05-15",
    )
    
    # embedding_function = OpenAIEmbeddings()
    
    vectorstore = Chroma(
        collection_name="bellavista",
        embedding_function=embedding_function,
        persist_directory="data",
    )

    docs = [
        Document(
            page_content=(
                "Bella Vista is a cozy Italian restaurant offering panoramic city views "
                "and an inviting atmosphere."
            )
        ),
        Document(
            page_content=(
                "Bella Vista serves only ONE specific, well known food. a classic wood-fired pizza "
                "prepared with fresh tomatoes, mozzarella, and basil. Bella vista does not serve any other food"
            )
        ),
        Document(
            page_content=(
                "Patrons often enjoy the sunset while dining; reservations are "
                "recommended for a window table."
            )
        ),
    ]

    vectorstore.add_documents(docs)
    print("Bella Vista documents added to Chroma!")
    return vectorstore


if __name__ == "__main__":
    load_bellavista_documents()
