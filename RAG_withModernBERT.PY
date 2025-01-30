# INSTALLING NECESSARY LIBRARIES
!pip install git+https://github.com/huggingface/transformers
!pip install sentence-transformers
!pip install datasets
!pip install -U weaviate-client


# LOADING THE DATASET
from datasets import load_dataset
ds = load_dataset("kdave/Indian_Financial_News")
# Keep only "content" columns from the dataset
train_ds = ds["train"].select_columns(["Content"])

#SELECT 100 rows
import random
# Set seed
random.seed(42)
# Shuffle the dataset and select the first 100 rows
subset_ds = train_ds.shuffle(seed=42).select(range(100))



#EMBEDDINGS GENERATION WITH MODERNBERT-EMBED-BASE
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer("nomic-ai/modernbert-embed-base")

# Function to generate embeddings for a single text
def generate_embeddings(example):
    example["embeddings"] = model.encode(example["Content"])
    return example

# Apply the function to the dataset using map
embeddings_ds = subset_ds.map(generate_embeddings)


#  CONVERT HUGGING FACE DATASET TO A PANDAS DATAFRAME
import pandas as pd
# Convert HF dataset to Pandas DF
df = embeddings_ds.to_pandas()

#INSERTING THE EMBEDDINGS INTO WEVIATE
import weaviate
# Connect to Weaviate
client = weaviate.connect_to_embedded()

#CREATING A WEVIATE COLLECTION AND APPENDING THE EMBEDDINGS
import weaviate.classes as wvc
import weaviate.classes.config as wc
from weaviate.classes.config import Property, DataType

# Define the collection name
collection_name = "news_india"

# Delete the collection if it already exists
if (client.collections.exists(collection_name)):
    client.collections.delete(collection_name)

# Create the collection
collection = client.collections.create(
    collection_name,
    vectorizer_config = wvc.config.Configure.Vectorizer.none(),

    # Define properties of metadata
    properties=[
        wc.Property(
            name="Content",
            data_type=wc.DataType.TEXT
        )
      
    ]
)


objs = []
for i, d in enumerate(df["Content"]):
    objs.append(wvc.data.DataObject(
            properties={
                "Content": df["Content"][i]
            },
            vector = df["embeddings"][i].tolist()
        )
    )

collection.data.insert_many(objs)


#QUERYING THE DATA
query = "Which biscuits is Britannia Industries Ltd is looking at reducing prices for?"
top_n = 1

#Encoding the Query
from weaviate.classes.query import MetadataQuery
query_embedding = model.encode(
    query,
    convert_to_tensor=True
)

# PRINTING THE RETRIEVED RESULTS
print(f"Top {top_n} results:\n")
for i, obj in enumerate(results.objects):
    print(obj.properties['content'])
    print("\n")
    print("NEW LINE")
