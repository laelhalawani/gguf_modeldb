# glai
GGUF LLAMA AI - Package for simplified text generation with Llama models quantized to GGUF format

Provides high level APIs for loading models and generating text completions.
Visit API documentation at: https://laelhalawani.github.io/glai/

## High Level API Classes:

### AutoAI:
- Automatically searches for and loads a model based on name/quantization/keyword. 
- Handles downloading model data, loading it to memory, and configuring message formatting.
- Use generate() method to get completions by providing a user message.

### EasyAI:
- Allows manually configuring model data source - from file, URL, or ModelDB search.
- Handles downloading model data, loading it to memory, and configuring message formatting.
- Use generate() method to get completions by providing a user message.

### ModelDB (used by AutoAI and EasyAI):
- Manages database of model data files. via ModelData class objects.
- Useful for searching for models and retrieving model metadata.
- Can import models from HuggingFace repo URLs or import and download models from .gguf urls on huggingface.

### ModelData (used by ModelDB):
- Represents metadata and info about a specific model.
- Used by ModelDB to track and load models.
- Can be initialized from URL, file, or ModelDB search.
- Used by ModelDB to download model gguf file 

## Installation
To install the package use pip
```
pip install glai
```
## Usage:
Usage examples. 

### Import package
```python
from glai import AutoAI, EasyAI, ModelDB, ModelData
```
### AutoAI - automatic model loading
```python
ai = AutoAI(name_search="Mistral")
ai.generate("Hello") 
```
### EasyAI - manual model configuration 
```python
easy = EasyAI()
easy.load_model_db()
easy.find_model_data(name_search="Mistral")
easy.load_ai()
easy.generate("Hello")
```
### ModelDB - search models and show db info
```python
from llgg import ModelDB
db = ModelDB()
model = db.find_model(name_search="Mistral")
print(model.name)
db.show_db_info()
```
### Import Models from Repo
Import models from a HuggingFace repo into the model database:
```python
from glai.back_end.model_db.db import ModelDB

mdb = ModelDB('./gguf_db', False)
mdb.import_models_from_repo(
    hf_repo_url="https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF",
    user_tags=["[INST]", "[/INST]"],
    ai_tags=["", ""],
    description="We introduce SOLAR-10.7B, an advanced large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. It's compact, yet remarkably powerful, and demonstrates unparalleled state-of-the-art performance in models with parameters under 30B.",
    keywords=["10.7B", "upstage","isntruct", "solar"],
    replace_existing=False,
)
mdb.show_db_info()
```
### AutoAI Quick Example
Quickly generate using AutoAI:
```python
from glai.ai import AutoAI

auto_ai = AutoAI("zephyr", "q2_k", max_total_tokens=100)
auto_ai.generate(
    user_message_text="Output just 'hi' in single quotes with no other prose. Do not include any additional information nor comments.",
    ai_message_to_be_continued= "'",
    stop_at="'",
    include_stop_str=True
)
```
### EasyAI Step By Step Example
Step by step generation with EasyAI:

```python
from glai.ai import EasyAI

easy_ai = EasyAI()
easy_ai.load_model_db('./gguf_db')
easy_ai.find_model_data("zephyr", "q2_k")
easy_ai.load_ai()
easy_ai.generate(
    "Output a list of 3 strings. The first string should be `hi`, the second string should be `there`, and the third string should be `!`.",
    "['",
    "']"
)
```
### EasyAI All In One Example
All in one generation with EasyAI:

```python
from glai.ai import EasyAI

easy_ai = EasyAI()
easy_ai.configure(
    model_db_dir="./gguf_db",
    name_search="zephyr",
    quantization_search="q2_k",
    max_total_tokens=100
)
easy_ai.generate(
    "Output a python list of 3 unique cat names.", 
    "['", 
    "']"
)
```
### AutoAI from Dict Example
Generate from AutoAI using a config dict:
```python
from glai.ai import AutoAI

conf = {
  "model_db_dir": "./gguf_db",
  "name_search": "zephyr",
  "quantization_search": "q2_k",
  "keyword_search": None,
  "max_total_tokens": 300 
}

AutoAI(**conf).generate(
  "Please output only the provided message as python list.\nMessage:`This string`.",
  "['", 
  "]", 
  True
)
```
### EasyAI from Dict Example
Generate from EasyAI using a config dict:

```python
from glai.ai import EasyAI

conf = {
  "model_db_dir": "./gguf_db",
  "name_search": "zephyr",
  "quantization_search": "q2_k",
  "keyword_search": None,
  "max_total_tokens": 300,
}

EasyAI(**conf).generate(
  "Please output only the provided message as python list.\nMessage:`This string`.",
  "['",
  "']",
  True  
)
```
### EasyAI from URL Example
Get a model from a URL and generate:

```python
from glai.back_end.model_db.db import ModelDB
from glai.ai import EasyAI

mdb = ModelDB('./gguf_db', False)
mdb.show_db_info()

eai = EasyAI()
eai.load_model_db('./gguf_db')
eai.model_data_from_url(
    url="https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/blob/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    user_tags=("[INST]", "[/INST]"),
    ai_tags=("", ""),
    description="The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mistral-8x7B outperforms Llama 2 70B on most benchmarks we tested.",
    keywords=["mixtral", "8x7b", "instruct", "v0.1", "MoE"],
    save=True,
)
eai.load_ai(max_total_tokens=300)
eai.generate(
    user_message="Write a short joke that's actually super funny hilarious best joke.",
    ai_response_content_tbc="",
    stop_at=None,
    include_stop_str=True,
)
```
Detailed API documentation can be found here: https://laelhalawani.github.io/glai/