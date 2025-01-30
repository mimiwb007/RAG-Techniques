# INSTALL NECESSARY LIBRARIES
!pip install lightrag-hku
!pip install aioboto3
!pip install tiktoken
!pip install nano_vectordb

#Install Ollama
!sudo apt update
!sudo apt install -y pciutils
!pip install langchain-ollama
!curl -fsSL https://ollama.com/install.sh | sh
!pip install ollama==0.4.2



#IMPORT NECESSARY LIBRARIES & DEFINE OPEN AI KEY
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
import os
os.environ['OPENAI_API_KEY'] =''
import nest_asyncio
nest_asyncio.apply()
WORKING_DIR = "./content"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


#CALLING THE TOOL & LOADING THE DATA
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

#Insert Data
with open("./Coffee.txt") as f:
    rag.insert(f.read())


# QUERYING ON  QUESTION
print(rag.query("Which section of Indian Society is Coffee getting traction in?", param=QueryParam(mode="hybrid")))
print(rag.query("Which section of Indian Society is Coffee getting traction in?", param=QueryParam(mode="naive")))
