import os
import json
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

IS_CI = os.getenv("CI") == "true"


def load_chatdoctor_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for row in data:
        text = f"""
Instruction: {row.get('instruction', '')}
Input: {row.get('input', '')}
Output: {row.get('output', '')}
""".strip()
        docs.append(Document(page_content=text))

    return docs


def create_rag_pipeline(build_index: bool = False):
    # ✅ Skip FAISS creation in normal CI
    if IS_CI and not build_index:
        print("⚠️ CI detected → skipping RAG pipeline initialization")
        return None

    print("✅ Initializing RAG pipeline")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True},
    )

    if os.path.exists("faiss_store"):
        print("✅ Loading existing FAISS index")
        vectordb = FAISS.load_local(
            "faiss_store",
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        print("🔨 Building FAISS index")

        docs = load_chatdoctor_json("data/chatdoctor5k.json")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=60,
        )
        chunks = splitter.split_documents(docs)

        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local("faiss_store")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.7,
        max_output_tokens=4000,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2},
    )

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""
You are a medical consultant.
Use ONLY the provided dataset.
If the answer is not found, say:
"The answer is not available in the provided context."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
""",
    )

    print("✅ RAG pipeline ready")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer",
    )