# gguf_modeldb 

This package is a quick and optimized solution to manage llama based gguf quantized models,
download gguf files, retreive messege formatting, add more models from hf repos and more.
It's super easy to use via it's main `ModelDB` smart class and comes prepacked with the best open source models including:
dolphin phi-2 2.7b, mistral 7b v0.2, mixtral 8x7b v0.1, solar 10.7b and zephyr 3b
with all their available quantizations (from q2_k to q8_0)
and with correct assistant, user and system message tags configured.

## Get started

### Installation
The package is available on PyPI
```bash
pip install gguf_modeldb
```
### Find and download a gguf model
```python
from gguf_modeldb import ModelDB
mdb = ModelDB()
mdb.show_db_info() #prints nice summary of all available models
model_data = mdb.find_model("dolphin", "q2_k") #find model data for dolphin smallest quant

#inspect the data
print(f"""
        Model Name: {mdt.name}
        Model Quantization: {mdt.model_quantization}
        Model Path: {mdt.model_path()}
        Model message formattings:
            System: {mdt.system_tags}
            User: {mdt.user_tags}
            AI: {mdt.ai_tags}
        Description: {mdt.description}
        Keywords: {mdt.keywords}
        Is downloaded: {mdt.is_downloaded()}
    """)
#outputs:
    # Model Name: dolphin-2_6-phi-2
    # Model Quantization: Q2_K
    # Model Path: your_db_dir_path\dolphin-2_6-phi-2.Q2_K.gguf    
    # Model message formattings:
    #     System: {'open': '<|im_start|>system\n', 'close': '<|im_end|>\n'}
    #     User: {'open': '<|im_start|>user\n', 'close': '<|im_end|>\n'}
    #     AI: {'open': '<|im_start|>assistant', 'close': '<|im_end|>\n'}
    # Description: Dolphin 2.6 phi 2 GGUF
    # Keywords: ['dolphin', 'phi2', 'uncesored', '2.7B']
    # Is downloaded: False #unlsess you downloaded it already

#if you want you can easily download the model
model_data.download_gguf()
#now it can be passed to llama-cpp or gguf_llama for inference
gguf_path = model_data.model_path()

#importantly if you use the default model db dir, you can reuse these models across any project
#without the need to separately download and configure them, you download each model only once 
#when you try to use that specific version of the model for the first time
```
### API documentation
Detailed api documentation available at https://laelhalawani.github.io/gguf_modeldb
# Examples
Below you will find some more examples from examples.py in the model repository.

## Import ModelDB class
```python
from gguf_modeldb import ModelDB
```
## Show available models
Display information about all models (similar to the model summary at the end of this readme)
```python
def show_available_models():
    mdb = ModelDB()
    mdb.show_db_info()
```
## Use model db super easily
All you need to have the model data and/or download the model gguf and get it's file path
is available in two lines.
This example initializes the default ModelDB (without providing alternative db dir as an argument).
Then it searchers entries for 'dolpin' 'q2_0' and returns the best matched entry.
Finally you access the data and display it's detailes and download the model.
```python
def use_default_db_easily():
    mdb = ModelDB()
    mdt = mdb.find_model('dolphin', 'q2_0')
    print(f"""
            Model Name: {mdt.name}
            Model Quantization: {mdt.model_quantization}
            Model Path: {mdt.model_path()}
            Model message formattings:
                System: {mdt.system_tags}
                User: {mdt.user_tags}
                AI: {mdt.ai_tags}
            Description: {mdt.description}
            Keywords: {mdt.keywords}
            Is downloaded: {mdt.is_downloaded()}
        """)
    mdt.download_gguf()
    #to do it in one line
    #ModelDB().find_model('dolphin','q2_0').download_gguf()
```
## Import verified model into your model db
Set up a model db in a provided directory.
Then try to find and import a model 'dolphin' with 'q2_0' quantization from the built in json model db that comes with the package. It will copy the model data json file into your dir.
Next your dtb is searched for the imported model and the details of the model are listed.
Finally the model gguf file is downloaded to the directory.

```python
def custom_db_dir_and_model_import(your_db_dir:str):
    mdb = ModelDB(your_db_dir, False)
    mdb.import_verified_model('dolhpin', 'q2_0')
    mdt = mdb.find_model('dolphin', 'q2_0')

    print(f"""
        Model Name: {mdt.name}
        Model Quantization: {mdt.model_quantization}
        Model Path: {mdt.model_path()}
        Model message formattings:
            System: {mdt.system_tags}
            User: {mdt.user_tags}
            AI: {mdt.ai_tags}
        Description: {mdt.description}
        Keywords: {mdt.keywords}
        Is downloaded: {mdt.is_downloaded()}
    """)
    mdt.download_gguf()
```

## Import more models from HF easy!
With gguf_modeldb you can easily extend your collection of models.
You can save the information from the HF repo by providing a link,
which will create json entries in the active model db dir. 
Make sure to check what format of tags (if any) is the model fine tuned with. 
And if it supports system prompts make sure to provide these tags too. 
The other arguments description and keywords are optional and helpful for finding the model in your db later.
Finally if you want to overwrite existing models data from this repo (because i.e. you chose wrong tags) you can set the last parameter to `True`
    
```python
def add_gguf_model_repo_from_huggingface(your_db_dir:str):
    #The url needs to lead to a website that lists url links ending with .gguf
    #TheBloke's repositories usually provide a table with such links
    #These links are detected and added and for each model data is built and added to the db
    #Nothing is downloaded, only the model data is added to the db
    #You can initialize the download later with .download_gguf() on model data object return from find_model
    hf_repo_url:str = "https://huggingface.co/TheBloke/CodeLlama-7B-GGUF"
    mdb = ModelDB(your_db_dir, False) #False means don't copy all model data from default db to your db
    mdb.import_models_from_repo(
        hf_repo_url=hf_repo_url,
        user_tags=["",""],
        ai_tags=["",""],
        system_tags=None,
        description="Code Llama is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 34 billion parameters. This is the repository for the base 7B version in the Hugging Face Transformers format. This model is designed for general code synthesis and understanding.",
        keywords=["Llama2", "Meta", "Code", "Completions"],
        replace_existing=True #if model data jsons already exist in your db, replace them with the new ones
    )

```

# Model Summary
gguf_modeldb comes prepacked with over 50 preconfigured, ready to download and deploy model x quantization versions from verified links on huggingface, with configured formatting data allowing you to download 
and get all model data in one line of code, then just pass it to llama-cpp-python or gguf_llama instance
for much smoother inference. Below is the summary of the available models. 

**Number of models:** 56

## Available Models:

### dolphin-2_6-phi-2:
- **Quantizations:** ['Q2_K', 'Q3_K_L', 'Q3_K_M', 'Q3_K_S', 'Q4_0', 'Q4_K_M', 'Q4_K_S', 'Q5_0', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0']
- **Keywords:** ['dolphin', 'phi2', 'uncensored', '2.7B']
- **Description:** Dolphin 2.6 phi 2 GGUF

---

### mistral-7b-instruct-v0.2:
- **Quantizations:** ['Q2_K', 'Q3_K_L', 'Q3_K_M', 'Q3_K_S', 'Q4_0', 'Q4_K_M', 'Q4_K_S', 'Q5_0', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0']
- **Keywords:** ['Mistral', '7B', 'INST', 'v0.2', 'default', 'instruct', 'uncensored', 'open-source', 'apache']
- **Description:** The Mistral-7B-Instruct-v0.2 Large Language Model (LLM) is an improved instruct fine-tuned version of Mistral-7B-Instruct-v0.1.

---

### mixtral-8x7b-instruct-v0.1:
- **Quantizations:** ['Q2_K', 'Q3_K_M', 'Q4_0', 'Q4_K_M', 'Q5_0', 'Q5_K_M', 'Q6_K', 'Q8_0']
- **Keywords:** ['mixtral', '8x7b', 'instruct', 'v0.1', 'MoE']
- **Description:** The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mistral-8x7B outperforms Llama 2 70B on most benchmarks we tested.

---

### solar-10.7b-instruct-v1.0:
- **Quantizations:** ['Q2_K', 'Q3_K_L', 'Q3_K_M', 'Q3_K_S', 'Q4_0', 'Q4_K_M', 'Q4_K_S', 'Q5_0', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0']
- **Keywords:** ['10.7B', 'upstage', 'instruct', 'solar']
- **Description:** We introduce SOLAR-10.7B, an advanced large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. It's compact, yet remarkably powerful, and demonstrates unparalleled state-of-the-art performance in models with parameters under 30B.

---

### stablelm-zephyr-3b:
- **Quantizations:** ['Q2_K', 'Q3_K_L', 'Q3_K_M', 'Q3_K_S', 'Q4_0', 'Q4_K_M', 'Q4_K_S', 'Q5_0', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0']
- **Keywords:** ['zephyr', '3b', 'instruct', 'non-commercial', 'research']
- **Description:** StableLM Zephyr 3B is a 3 billion parameter instruction tuned inspired by HugginFaceH4's Zephyr 7B training pipeline. This model was trained on a mix of publicly available datasets, synthetic datasets using Direct Preference Optimization (DPO). Evaluation for this model is based on MT Bench and Alpaca Benchmark.

---

# Contributions
All contributions are welcome, please feel encouraged to send your PRs on develop branch.
