import tiktoken
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


embeddings = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY")

local_index_filename = "faiss_index"
pdf_filename = "./data/data.pdf"


def get_new_vector_db_and_store_local_index(
        base_filename=pdf_filename,
        index_name=local_index_filename,
):
    # load the document and split it into chunks
    loader = PyPDFLoader(base_filename)
    pages = loader.load_and_split()

    # split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=tiktoken_len  # tictoken_len은 text를 나눌때 어떤 기준으로 할것인지에 대한 결과값을 제공합니다. 저는 openai에서 사용하는 기준으로 했습니다.
    )
    docs = text_splitter.split_documents(pages)  # docs는 나눠진 문서들에 배열입니다.

    new_vectordb = FAISS.from_documents(docs, embeddings)
    new_vectordb.save_local(index_name)  # index 정보를 local에 저장합니다.

    return new_vectordb


# if you can find local_index_file, load_local
try:
    vectordb = FAISS.load_local(local_index_filename, embeddings)
except RuntimeError:
    print("failed to load FAISS")
    print("crete new local index")
    vectordb = get_new_vector_db_and_store_local_index(base_filename=pdf_filename, index_name=local_index_filename)

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 10})

question = "클라우드 네이티브 앱 모범 사례에 대해서 말해줘."
docs_from_db = retriever.invoke(question)
print(docs_from_db[0].metadata)
# print(docs_from_db[1].metadata)
# print(docs_from_db[2].metadata)

docs_and_scores = vectordb.similarity_search_with_score(question)
print(docs_and_scores)

llm = ChatOpenAI(openai_api_key="OPENAI_API_KEY")

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": question})
print(response["input"])
print(response["context"])
print(response["answer"])
