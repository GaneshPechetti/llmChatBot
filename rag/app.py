import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_models import ChatCohere
from streamlit.external.langchain import StreamlitCallbackHandler
cohere_api_key = os.environ.get("COHERE_API_KEY")
llm = ChatCohere(
        model="command",
        max_tokens=256,
        temperature=0.5, 
        cohere_api_key=cohere_api_key)

import logging
# setup logger
logging.basicConfig(encoding='utf-8', level=logging.INFO)

# import relevant chains
from langchain.chains import (
    ConversationalRetrievalChain,
    FlareChain,
    SimpleSequentialChain
)

# set up document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader
)

supported_extensions = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".epub": UnstructuredEPubLoader,
    ".doc": UnstructuredWordDocumentLoader
}


from langchain.schema import Document, BaseRetriever
from langchain_community.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
EMBEDDER = CohereEmbeddings(model="embed-english-light-v3.0")

# create memory
from langchain.memory import ConversationBufferMemory
def init_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

MEMORY = init_memory()

# set up a retriever
def configure_retriever(
        docs: list[Document],
        use_compression: bool = False
) -> BaseRetriever:
    # Split each document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    logging.info(splits)

    # Extract text content from each document
    serialized_docs = []
    for doc in splits:
        serialized_doc = {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }
        serialized_docs.append(serialized_doc)

    # Extract the text content from each document
    texts = [doc['page_content'] for doc in serialized_docs]

    # Embed the documents using Cohere
    embeddings = EMBEDDER.embed_documents(texts)

    # Store them into the vector database
    db = Chroma.from_documents(splits, EMBEDDER)

    # Configure a retriever
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={
            "k": 5,
            "fetch_k": 7,
            "include_metadata": True
        }
    )

    # Return based on the compression flag
    if not use_compression:
        return retriever

    efilter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.5
    )

    return ContextualCompressionRetriever(
        base_compressor=efilter,
        base_retriever=retriever
    )



from langchain.chains.base import Chain
import pathlib


def load_document(temp_filepath: str) -> list[Document]:
    """load a file and return it as list of documents / chunks"""
    ext = pathlib.Path(temp_filepath).suffix
    loader = supported_extensions.get(ext)
    if not loader:
        raise ValueError(f"Document {ext} type not supported")
    loaded = loader(temp_filepath)
    docs = loaded.load()
    logging.info(docs)
    return docs

def configure_chain(
        retriever: BaseRetriever,
        use_flare: bool = True
) -> Chain:
    params = dict(
        llm = llm,
        retriever = retriever,
        memory = MEMORY,
        verbose = True,
        max_tokens_limit=4000
    )

    if use_flare:
        return FlareChain.from_llm(**params)
    
    return ConversationalRetrievalChain.from_llm(**params)


def configure_retrieval_chain(
        uploaded_files,
        use_compression: bool=False,
        use_flare: bool = False,
) -> Chain:
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriever(docs=docs, use_compression=use_compression)
    chain = configure_chain(retriever=retriever, use_flare=use_flare)
    return chain




# layout of the application
st.set_page_config("RAG usig Langchain", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")

uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    type = list(supported_extensions.keys()),
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload documents to the knowledge base")
    st.stop()

# add checkboxes
use_compression = st.checkbox("compression", value=False)
use_flare = st.checkbox("flare", value=False)

CONV_CHAIN = configure_retrieval_chain(
    uploaded_files,
    use_compression=use_compression,
    use_flare=use_flare
)

if st.sidebar.button("Cleat message history"):
    MEMORY.chat_memory.clear()

agents = {"user": "user", "chatbot":"assistant"}

if len(MEMORY.chat_memory.messages)==0:
    st.chat_message("assistant").markdown("Ask me anything")

for msg in MEMORY.chat_memory.messages:
    st.chat_message(agents[msg.type].write(msg.content))

assistant = st.chat_message("assistant")

if user_query := st.chat_input(placeholder="Summarize the documents"):
    st.chat_message("user").write(user_query)
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)
    with st.chat_message("assistant"):
        response = CONV_CHAIN.invoke(
            input = {
                "question": user_query,
                "chat_history": MEMORY.chat_memory.messages
            }, callbacks = [stream_handler]
        )

        if response:
            container.markdown(response['answer'])

