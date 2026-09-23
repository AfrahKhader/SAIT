# """Retrieval infrastructure: builds/loads the Chroma store and its tool.

from __future__ import annotations

import glob
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


class KnowledgeBase:
    """Owns the embeddings, vector store, and retriever for the lecture notes."""

    def __init__(self, config: Config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(model=config.llm.embedding_model)
        self.vectorstore: Chroma | None = None
        self.retriever = None

    def build(self) -> "KnowledgeBase":
        """Prepare the vector store and retriever. Returns self for chaining."""
        self.vectorstore = self._get_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type=self.config.retriever.search_type,
            search_kwargs={"k": self.config.retriever.k},
        )
        return self

    def as_retriever(self):
        if self.retriever is None:
            raise RuntimeError("KnowledgeBase.build() must be called first.")
        return self.retriever

    # -- ------------------internals -------------------------------------

    def _get_vectorstore(self) -> Chroma:
        vs = self.config.vectorstore
        already_built = os.path.isdir(vs.persist_directory) and os.listdir(
            vs.persist_directory
        )

        if already_built and not vs.rebuild:
            print("Loading existing ChromaDB vector store...")
            return Chroma(
                persist_directory=vs.persist_directory,
                collection_name=vs.collection_name,
                embedding_function=self.embeddings,
            )

        print("Building ChromaDB vector store from PDFs...")
        chunks = self._load_and_split()
        os.makedirs(vs.persist_directory, exist_ok=True)
        store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=vs.persist_directory,
            collection_name=vs.collection_name,
        )
        print(f"Created ChromaDB vector store with {len(chunks)} chunks!")
        return store

    def _load_and_split(self):
        docs_cfg = self.config.documents
        pdf_paths = glob.glob(os.path.join(docs_cfg.pdf_dir, "*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in: {docs_cfg.pdf_dir}")

        all_pages = []
        for path in pdf_paths:
            try:
                pages = PyPDFLoader(path).load()
                print(f"Loaded {os.path.basename(path)} — {len(pages)} pages")
                all_pages.extend(pages)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                raise

        print(f"Total pages across all PDFs: {len(all_pages)}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=docs_cfg.chunk_size,
            chunk_overlap=docs_cfg.chunk_overlap,
        )
        return splitter.split_documents(all_pages)


def make_retriever_tool(retriever):
    """Return a LangChain tool that searches the lecture notes."""

    @tool
    def retriever_tool(query: str) -> str:
        """Search and return information from the lecture note documents."""
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the lecture notes document."

        results = []
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            results.append(
                f"Document {i + 1} (source: {source}, page: {page}):\n{doc.page_content}"
            )
        return "\n\n".join(results)

    return retriever_tool