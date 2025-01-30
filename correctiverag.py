# Installing the Necessary Libraries
!pip install tiktoken langchain-openai langchainhub chromadb langchain langgraph tavily-python
!pip install -qU pypdf langchain_community

# DEFINING THE API KEYS
import os
os.environ["TAVILY_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

# IMPORTING THE NECESSARY LIBRARIES
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START

# CHUNKING THE DOCUMENT AND CREATING THE RETRIEVER
file_path = "Brochure_Basic-Creative-coffee-recipes.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()



#  SETTING UP THE RAG CHAIN
# Prompt
rag_prompt = hub.pull("rlm/rag-prompt")
# LLM
rag_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Chain
rag_chain = rag_prompt | rag_llm | StrOutputParser()
print(rag_prompt.messages[0].prompt.template)



# SETTING UP AN EVALUATOR
### Retrieval Evaluator
class Evaluator(BaseModel):
    """Classify retrieved documents based on its relevance to the question."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    
# LLM with function call
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_llm_evaluator = evaluator_llm.with_structured_output(Evaluator)

# Prompt
system = """You are tasked with evaluating the relevance of a retrieved document to a user's question. \n If the document contains keywords or semantic content related to the question, mark it as relevant. \n Output a binary score: 'yes' if the document is relevant, or 'no' if it is not"""
retrieval_evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \\n\\n {document} \\n\\n User question: {question}"),
    ]
)
retrieval_grader = retrieval_evaluator_prompt | output_llm_evaluator 


# SETTING UP A QUERY REWRITER
question_rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Prompt
system = """You are a question rewriter who improves input questions to make them more effective for web search. \n Analyze the question and focus on the underlying semantic intent to craft a better version."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \\n\\n {question} \\n Formulate an improved question.",
        ),
    ]
)
question_rewriter = re_write_prompt | question_rewriter_llm | StrOutputParser()


# SETTING UP WEB SEARCH
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)



# SETTING UP THE GRAPH STATE FOR LANGGRAPH
class GraphState(TypedDict):
    
    question: str
    generation: str
    web_search: str
    documents: List[str]



# SETTING UP THE FUNCTION NODES 
# Function for Retrieving Relevant Documents
def retrieve(state):    
    question = state["question"]
    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

# Function for Generating Answers From the Retrieved Documents
def generate(state):    
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# Function for Evaluating the Retrieved Documents
def evaluate_documents(state):
  
    question = state["question"]
    documents_all = state["documents"]
    # Score each doc
    docs_filtered = []
    web_search = "No"
    
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            docs_filtered .append(d)
        else:
            continue
    
    if len(docs_filtered) / len(documents_all) <= 0.7:
        web_search = "Yes"
    return {"documents": docs_filtered, "question": question, "web_search": web_search}

#Function for Transforming the User Query For Better Retrieval
def transform_query(state):    
    question = state["question"]
    documents_all = state["documents"]
    # Re-write question
    transformed_question = question_rewriter.invoke({"question": question})
    return {"documents": documents_all , "question": transformed_question }


#Function for Web Searching
def web_search(state):    
    question = state["question"]
    documents_all = state["documents"]
    # Web search
    docs = web_search_tool.invoke({"query": question})
    
    #Fetch results from web
    web_results = "\\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    #Append the results from the web to the documents
    documents_all.append(web_results)
    return {"documents": documents_all, "question": question}

#Function for Deciding Next Step - whether to generate or transforming the query for web search
def decide_next_step(state):   
    web_search = state["web_search"]
    if web_search == "Yes":   
         return "transform_query"
    else:
        return "generate"



# CONNECTING ALL THE FUNCTION NODES & ADDING EDGES
workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", evaluate_documents)  # evaluate documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search_node", web_search)  # web search

# Adding the Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)
# Compile
app = workflow.compile()


#  OUTPUT USING CORRECTIVE RAG
from pprint import pprint
inputs = {"question": "What is the difference between Flat white and cappuccino?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        pprint(value, indent=2, width=80, depth=None)
    pprint("\\n---\\n")
# Final generation
pprint(value["generation"])
