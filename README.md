# RAG-Project-using-LLAMA
Harnessing the LLAMA framework, this project constructs a dynamic information retrieval system, adept at swiftly delivering precise responses and extracting insights from document repositories using advanced NLP techniques.

**Project Introduction:**

The project revolves around leveraging the LLAMA (Language Learning for Machines) framework to build an information retrieval system. This system aims to efficiently handle queries and provide relevant responses based on a corpus of documents. In this project, we'll walk through the implementation steps using a Google Colab notebook.

**Notebook Explanation:**

1. **Mounting Google Drive**: Initially, we mount Google Drive to access necessary files and directories.
2. **Installation of Dependencies**: Required dependencies such as LLAMA Index, OpenAI, PyPDF, and Python-dotenv are installed to set up the environment.
3. **Loading Environment Variables**: We load environment variables using the `dotenv` library, particularly focusing on `OPENAI_API_KEY`.
4. **Data Loading**: Documents are loaded from a specified directory using `SimpleDirectoryReader`.
5. **Index Creation**: A vector store index is created from the loaded documents. The index is then used to construct a query engine.
6. **Query Execution**: Queries regarding "YOLO" are executed using the query engine.
7. **Response Presentation**: Responses to the queries are formatted and presented, along with their sources.

The code is divided into logical sections, each focusing on a specific aspect of the information retrieval process. It starts with environment setup and data loading, followed by index creation and query execution.

**Code and Output:**

Now, let's delve into the code and corresponding outputs.

```python
# Mount Google Drive and set directory
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/LLM Project

# Check contents
!ls

# Check Python version
!python --version

# Install dependencies
!pip install llama-index
!pip install openai
!pip install pypdf
!pip install python-dotenv

# Import necessary libraries
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Load environment variables
var = os.environ['VAR_NAME']
print(var)

# Load data from directory
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader("data").load_data()

# Display loaded documents
documents

# Create index from documents
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# Display index
index

# Set up query engine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.80)
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

# Execute queries
response = query_engine.query("what is yolo?")

# Display responses
from llama_index.core.response.pprint_utils import pprint_response
pprint_response(response, show_source=True)
print(response)
```

Here is the code snippet, where the index is stored in the device for future use:

1. **Imports**: Necessary modules are imported, including `os.path` for filesystem operations and various components from `llama_index.core`.

2. **Checking Storage**: The existence of a storage directory is verified. If it doesn't exist, it indicates that the index hasn't been created yet.

3. **Index Creation**: If no storage directory is found, the script loads documents from a specified directory using `SimpleDirectoryReader` and creates an index (`VectorStoreIndex`) from these documents. The index is then stored for future use.

4. **Querying**: After either creating or loading the index, it is converted into a query engine, and a sample query is executed to demonstrate its functionality.

5. **Output**: The response to the sample query is printed, showcasing the retrieval capabilities of the created or loaded index.

Overall, this code snippet demonstrates the workflow of creating or loading an index, preparing it for querying, and executing a sample query to retrieve relevant information.

```python
# Check for existing storage directory
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # Load documents and create index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Store index for later use
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Execute query on the index
query_engine = index.as_query_engine()
response = query_engine.query("What do you mean by vulnerability assessment?")
print(response)
```

**Output:**

- The output includes various responses to queries, along with their sources.
- Additionally, it demonstrates the loading and creation of an index, ensuring persistence for future use.


![O1](https://github.com/DionBenFernandes-Dev/RAG-Project-using-LLAMA/blob/main/Screenshot%202024-03-18%20121558.png)

![O2](https://github.com/DionBenFernandes-Dev/RAG-Project-using-LLAMA/blob/main/Screenshot%202024-03-18%20121703.png)
