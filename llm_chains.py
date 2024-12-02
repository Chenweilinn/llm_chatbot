from prompt_templates import memory_prompt_template
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers, LlamaCpp, HuggingFacePipeline
from transformers import pipeline
from langchain.vectorstores import Chroma
import chromadb
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def create_llm(model_path = config["model_path"]["large"], model_type = config["model_type"], model_config = config["model_config"]):
    # Load the LLM model
    # Load local model
    pipe = pipeline(
        "text-generation",
        model=model_path,    
        device=0  # For GPU, -1 for CPU
    )

    # Create LangChain instance
    llm = HuggingFacePipeline(pipeline=pipe)
   
    #llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm

def create_embeddings(embeddings_path = config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)
    
def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=5)  # k is the number of previous turns to consider


def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)


def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory) 

def load_normal_chain(chathistorty):
    return chatchain(chathistorty)

def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient("chroma_db")
    
    langchain_chroma = Chroma(   
        client =persistent_client,
        collection_name="pdfs",
        embedding_function=embeddings
        )
    
    return langchain_chroma

def load_pdf_chat_chain(chat_history):
    return pdfChatChain(chat_history)

def load_retrieval_chain(llm, memory, vector_db):
        
        return RetrievalQA.from_llm(llm=llm, memory=memory, retriever=vector_db.as_retriever(kwargs={"k": 3}))
class pdfChatChain:

    def __init__(self,chat_history):
        self.memory = create_chat_memory(chat_history)
        self.vector_db = load_vectordb(create_embeddings())
        llm =create_llm()
        #chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = load_retrieval_chain(llm,self.memory,self.vector_db)
        
    
    def run(self, user_input):
        
        return self.llm_chain.run(query =user_input, history= self.memory.chat_memory.messages, stop=["Human:"])

class chatchain:

    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm=llm, chat_prompt=chat_prompt, memory=self.memory) 

    def run(self, user_input):
        return self.llm_chain.run(human_input=user_input, history=self.memory.chat_memory.messages, stop=["Human:"]) # stop=["Human:"] is used to stop the model from generating Human questions itself

