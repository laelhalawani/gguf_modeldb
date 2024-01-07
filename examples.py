from gguf_modeldb import ModelDB

def show_available_models():
    mdb = ModelDB()
    mdb.show_db_info()

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