from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from redisvl.utils.vectorize import HFTextVectorizer

from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex

# load expects an iterable of dictionaries
from redisvl.redis.utils import array_to_buffer

from redisvl.query import VectorQuery

import os

# Redis value running in docker container
## docker run -d --name redis-stack -p 6379:6379 -e REDIS_ARGS="--requirepass p1234" redis/redis-stack-server:latest
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "p1234")

REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

# load list of pdfs from folder
data_path = "resources/"
docs = [os.path.join(data_path, file) for file in os.listdir(data_path)]

#print("Listing available documents ...", docs)

doc = [doc for doc in docs if "nke" in doc][0]

# set up the file loader/extractor and text splitter to create chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500, chunk_overlap=0
)

loader = UnstructuredFileLoader(
    doc, mode="single", strategy="fast"
)

# extract, load, and make chunks
chunks = loader.load_and_split(text_splitter)

print("Done processing. Created", len(chunks), "chunks of the original pdf", doc)

## Text embedding generation with RedisVL

hf = HFTextVectorizer("sentence-transformers/all-MiniLM-L6-v2")
os.environ["TOKENIZERS_PARRALELISM"] = "false"

#embedd each chunk content
embeddings = hf.embed_many([chunk.page_content for chunk in chunks])

# Check to make sure we've created enough embeddings, 1 per document chunk
print("Chunk Length = Embedding Length: ", len(embeddings) == len(chunks))

# Define a schema and create an index

index_name="redisvl"

schema = IndexSchema.from_dict({
    "index":{
        "name":index_name,
        "prefix":"chunk"
    },
    "fields":[{
        "name":"doc_id",
        "type":"tag",
        "attrs":{
            "sortable":True
        }
    },{
        "name":"context",
        "type":"text"
    },{
        "name":"text_embedding",
        "type":"vector",
        "attrs":{
            "dims": hf.dims,
            "distance_metric": "cosine",
            "algorithm":"hnsw",
            "datatype":"float32"
        }
    }
    ]
})

# connect to redis
client = Redis.from_url(REDIS_URL)

# create an index from schema and the client
index = SearchIndex(schema, client)
index.create(overwrite=True, drop=True)

print("Schema Index created")

## Process and load the dataset

# use redisvl index to load the list of document chunks into redis db
data = [
    {
    'doc_id': f'{i}',
    'content': chunk.page_content,
    # For Hash -- must convert embeddings to bytes
    'text_embedding': array_to_buffer(embeddings[i])
    } for i, chunk in enumerate(chunks)
]

# RedisVL Handles batching automatically
keys = index.load(data, id_field="doc_id")

## Query the database
query = "Nike profit margins and company performance"

query_embedding = hf.embed(query)

vector_query = VectorQuery(
    vector=query_embedding,
    vector_field_name="text_embedding",
    num_results=3,
    return_fields=["doc_id", "content"],
    return_score=True
)

# show the raw redis query
print(str(vector_query))

# execute the query with RedisVL
print(index.query(vector_query))
