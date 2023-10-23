from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
import dotenv

dotenv.load_dotenv()

# Step 1
raw_documents = TextLoader("./example.txt").load()

# Step 2
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20, length_function=len
)
documents = text_splitter.split_documents(raw_documents)

# Step 3
embeddings_model = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings_model)

# Step 4
retriever = db.as_retriever()

# Step 5
llm_src = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
qa_chain = create_qa_with_sources_chain(llm_src)
retrieval_qa = ConversationalRetrievalChain.from_llm(
    llm_src,
    retriever,
    return_source_documents=True,
)

# Output
output = retrieval_qa({
    "question": "What is the capital of France?",
    "chat_history": []
})
print(f"Question: {output['question']}")
print(f"Answer: {output['answer']}")
print(f"Source: {output['source_documents'][0].metadata['source']}")