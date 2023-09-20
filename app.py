
from typing import List
import os 

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document


import chainlit as cl
import gdown
import pathlib

from pathlib import Path
from tempfile import NamedTemporaryFile

import chainlit as cl
from chainlit.types import AskFileResponse
from chromadb.config import Settings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore




text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well.

Example of your response should be:

The answer is foo
SOURCES: xyz


Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}



def process_file(file):
    print(pathlib.Path(file).suffix)
    if pathlib.Path(file).suffix != ".pdf":
        raise TypeError("Only PDF files are supported")
    

    with NamedTemporaryFile() as tempfile:
        # tempfile.write(file.content)

        ######################################################################
        #
        print("Loading the file ",file)
        #
        ######################################################################
        p = Path.joinpath(Path.cwd(),file)
        print("Loading the file ",p)
        loader = PDFPlumberLoader(str(p))
        ######################################################################
        documents = loader.load()

        ######################################################################
        #
        # 2. Split the text
        #
        ######################################################################
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        ######################################################################

        docs = text_splitter.split_documents(documents)
        print(len(docs))
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
            
        return docs

def create_search_engine(files):
    final_docs=[]
    for file in files:
        docs = process_file(file=file)
        print("Filename is", file.name)
        print("This docs length is", len(docs))
        for doc in docs:
            final_docs.append(doc)

    print("Final docs length is", len(final_docs))


    
    ##########################################################################
    #
    # 3. Set the Encoder model for creating embeddings
    #
    ##########################################################################
    encoder = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )
    ##########################################################################

    # Save data in the user session
    cl.user_session.set("docs", final_docs)
    client_settings = Settings(
        # chroma_db_impl="duckdb+parquet",
        anonymized_telemetry=False,
        persist_directory=".chromadb",
        allow_reset=True
    )
    
    search_engine = Chroma(persist_directory=".chromadb",client_settings=client_settings)
    search_engine._client.reset()

    ##########################################################################
    #
    # 4. Create the document search engine. Remember to add 
    # client_settings using the above settings.
    #
    ##########################################################################
    search_engine = Chroma.from_documents(
        documents=final_docs,
        embedding=encoder,
        client_settings=client_settings,
        persist_directory=".chromadb" 
    )
    ##########################################################################

    return search_engine


@cl.on_chat_start
async def on_chat_start():
    files = None

    #download the files from the google drive 
    print("Downloading files from google drive")
    url=os.environ.get("GDRIVE_URL")
    print("URL from the env is ",{url})
    gdown.download_folder(url, quiet=True, use_cookies=False,remaining_ok=True)

    print("Downloaded files")

    files = [f for f in pathlib.Path('./Fintechdocs').iterdir() if f.is_file()]

    msg = cl.Message(
        content=f"Processing `{len(files)}` files", disable_human_feedback=True
    )
    await msg.send()


    print("Creating Search Engines")
    search_engine = create_search_engine(files)


    llm = ChatOpenAI(
        model='gpt-3.5-turbo-16k-0613',
        temperature=0,
        streaming=True
    )

    ##########################################################################
    #
    # 5. Create the chain / tool for RetrievalQAWithSourcesChain.
    #
    ##########################################################################
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=search_engine.as_retriever(max_tokens_limit=4097),
        ######################################################################
        # 6. Customize prompts to improve summarization and question
        # answering performance. Perhaps create your own prompt in prompts.py?
        ######################################################################
        chain_type_kwargs=chain_type_kwargs,
    )

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    print(message)
    res = await chain.acall(message, callbacks=[cb])
    
    print(res)
    answer = res["answer"]
    source_documents = res['sources']
    
    print(answer)
    print(source_documents)
    
    # text_elements = []  # type: List[cl.Text]

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         # Create the text element referenced in the message
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

    # print(source_names)
    #     if source_names:
    #         answer += f"\nSources: {', '.join(source_names)}"
    #     else:
    #         answer += "\nNo sources found"

    # await cl.Message(content=answer, elements=text_elements).send()
    await cl.Message(content=answer).send()
