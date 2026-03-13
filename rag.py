from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import (
    # OllamaEmbeddings,
    ChatOllama
    )
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# Required in env for local embedding model download `export HF_HUB_OFFLINE=1`


class Rag:
    __v_db: Chroma
    __doc_splitter: RecursiveCharacterTextSplitter
    __llm: ChatOllama

    def __init__(self):
        self.__v_db = Chroma(
            collection_name="tesing",
            embedding_function=HuggingFaceEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"trust_remote_code": True}
                ),
            # embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory="v_db/test_db"
        )
        self.__doc_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.__llm = ChatOllama(model="llama3.2:1b",
                                temperature=0.5,  # reducing randomness
                                top_k=1  # reducing randomness
                                )

    def load_to_vdb(self, directory_path: str):
        loader = DirectoryLoader(path=directory_path, glob="*.txt")
        docs = loader.lazy_load()
        doc_chunks = self.__doc_splitter.split_documents(docs)
        self.__v_db.add_documents(doc_chunks)

    def query_v_db(self, query: str):
        return self.__v_db.similarity_search(query, k=2)

    def generate_answer(self, question):
        prompt = ChatPromptTemplate.from_template(
            """Answer the question using the context below.

                Context:
                {context}

                Question:
                {input}
            """
        )
        question_answer_chain = create_stuff_documents_chain(self.__llm, prompt)
        chain = create_retrieval_chain(self.__v_db.as_retriever(), question_answer_chain)

        result = chain.invoke({"input": question})
        return result


rag = Rag()
# rag.load_to_vdb("data/")
print(rag.generate_answer("How much is the penalty for late maintenence?"))
