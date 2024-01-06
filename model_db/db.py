import bs4
import requests
from typing import Union, Optional

from util_helper.file_handler import create_dir, list_files_in_dir
from util_helper.compare_strings import compare_two_strings
from ..model_db import ModelData
from ..model_db.db_settings import MODEL_EXAMPLES_DB_DIR

class ModelDB:
    def __init__(self, model_db_dir:Optional[str]=None, copy_examples=True):
        self.gguf_db_dir = None
        self.models = []

        if model_db_dir is None:
            model_db_dir = MODEL_EXAMPLES_DB_DIR
        self.set_model_db_dir(model_db_dir)
        if model_db_dir != MODEL_EXAMPLES_DB_DIR:
            if copy_examples:
                print(f"Copying examples to {model_db_dir}...")
                for file in list_files_in_dir(MODEL_EXAMPLES_DB_DIR, include_directories=False, only_with_extensions=[".json"], just_names=False):
                    f_mdt = ModelData.from_json(file)
                    f_mdt.set_save_dir(model_db_dir)
                    f_mdt.save_json()
                    print(f"Saved a copy of {file} to {model_db_dir}.")
        else:
            print(f"Using default model db dir: {model_db_dir}, changes here will be visible and accessible globaly. If you would like to work with a specific db_dir provide it as an argument to the ModelDB constructor.")
        self.load_models()
    
    def set_model_db_dir(self, model_db_dir:str) -> None:
        print(f"ModelDB dir set to {model_db_dir}.")
        self.gguf_db_dir = create_dir(model_db_dir)
    
    def load_models(self) -> None:
        self.models = []
        files = list_files_in_dir(self.gguf_db_dir, include_directories=False, include_files=True, only_with_extensions=[".json"])
        for file in files:
            try:
                model_data = ModelData.from_json(file)
                self.models.append(model_data)
            except Exception as e:
                print(f"Error trying to load from {file}: \t\n{e}, \nskipping...")
                continue
        print(f"Loaded {len(self.models)} models from {self.gguf_db_dir}.")

    def find_models(self, name_query:Optional[str]=None, 
                   quantization_query:Optional[str]=None, 
                   keywords_query:Optional[str]=None,
                   treshold:float=0.5) -> Union[None, list]:
        if name_query is None and quantization_query is None and keywords_query is None:
            return None
        scoring_models_dict = {}
        for i, model in enumerate(self.models):
            scoring_models_dict[i] = {"model":model, "score":0}
        for id in scoring_models_dict.keys():
            model = scoring_models_dict[id]["model"]
            model:ModelData = model
            model_name = model.name
            model_quantization = model.model_quantization
            model_keywords = model.keywords
            if name_query is not None:
                #print(f"Searching for name: {name_query}")
                top_name_score = 0
                for model_subname in model_name.split("-"):
                    name_score = compare_two_strings(name_query, model_subname)
                    if name_score > top_name_score:
                        top_name_score = name_score
                if top_name_score > treshold:
                    scoring_models_dict[id]["score"] += top_name_score
                #print(f"Model {model_name} {model_quantization} top score: {top_name_score} treshold: {treshold}")
            if quantization_query is not None:
                #print(f"Searching for quantization: {quantization_query}")
                quantization_score = compare_two_strings(quantization_query, model_quantization)
                if quantization_score > treshold:
                    scoring_models_dict[id]["score"] += quantization_score
                #print(f"Model {model_name} {model_quantization} score: {quantization_score} treshold: {treshold}")
            if keywords_query is not None:
                #print(f"Searching for keyword: {keywords_query}")
                best_keyword_score = 0
                for keyword in model_keywords:
                    keyword_score = compare_two_strings(keywords_query, keyword)
                    if keyword_score > best_keyword_score:
                        best_keyword_score = keyword_score
                if best_keyword_score > treshold:
                    scoring_models_dict[id]["score"] += best_keyword_score
                #print(f"Model {model_name} {model_quantization} score: {best_keyword_score} treshold: {treshold}")
            #print(f"Model {model_name} {model_quantization} score: {scoring_models_dict[id]['score']}")
        sorted_models = sorted(scoring_models_dict.items(), key=lambda x: x[1]["score"], reverse=True)
        #keep just the list of model data
        sorted_models = [x[1]["model"] for x in sorted_models]
        #print(f"Found {len(sorted_models)} models.")
        #print(sorted_models)
        return sorted_models
    
    def find_model(self, name_query:Optional[str]=None, 
                   quantization_query:Optional[str]=None, 
                   keywords_query:Optional[str]=None,
                   ) -> Optional[ModelData]:
        sorted_models = self.find_models(name_query, quantization_query, keywords_query)
        if sorted_models is None:
            return None
        else:
            #print(f"Found {len(sorted_models)} models.")
            #print(sorted_models)
            return sorted_models[0]
    def get_model_by_url(self, url:str) -> Optional[ModelData]:
        for model in self.models:
            model:ModelData = model
            if model.gguf_url == url:
                return model
        return None    
    def add_model_data(self, model_data:ModelData, save_model=True) -> None:
        self.models.append(model_data)
        if save_model:
            model_data.save_json()
    
    def add_model_by_url(self, url:str, ) -> None:
        model_data = ModelData(url, db_dir=self.gguf_db_dir)
        self.add_model_data(model_data)

    def add_model_by_json(self, json_file_path:str) -> None:
        model_data = ModelData.from_json(json_file_path)
        self.add_model_data(model_data)

    def save_all_models(self) -> None:
        for model in self.models:
            model:ModelData = model
            model.save_json()
                
    @staticmethod
    def _model_links_from_repo(hf_repo_url:str):
        #extract models from hf 
        response = requests.get(hf_repo_url)
        html = response.text
        soup = bs4.BeautifulSoup(html, 'html.parser')
        #find all links that end with .gguf
        print(f"Looking for {hf_repo_url} gguf files...")
        model_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href is not None and href.endswith(".gguf"):
                print(f"Found model: {href}")
                model_links.append(href)
        return model_links
    def load_models_data_from_repo(self, hf_repo_url:str, 
                        user_tags:Optional[list[str]]=None,
                        ai_tags:Optional[list[str]]=None,
                        system_tags:Optional[list[str]]=None,
                        keywords:Optional[list[str]]=None, 
                        description:Optional[str]=None):  
        #create model data from hf repo
        model_links = ModelDB._model_links_from_repo(hf_repo_url)
        model_datas = []
        for model_link in model_links:
            model_data = ModelData(gguf_url=model_link, db_dir=self.gguf_db_dir, user_tags=user_tags, ai_tags=ai_tags, system_tags=system_tags, description=description, keywords=keywords)
            model_datas.append(model_data)
            model_data.save_json()
        self.models.extend(model_datas)
        return model_datas

    def import_models_from_repo(self, hf_repo_url:str,
                        user_tags:Optional[list[str]]=None,
                        ai_tags:Optional[list[str]]=None,
                        system_tags:Optional[list[str]]=None,
                        keywords:Optional[list[str]]=None, 
                        description:Optional[str]=None,
                        replace_existing:bool=False,
                        ):  
        #create model data from hf repo
        model_links = ModelDB._model_links_from_repo(hf_repo_url)
        print(f"Loaded {len(model_links)} models from {hf_repo_url}.")
        for model_link in model_links:
            model_data = ModelData(gguf_url=model_link, db_dir=self.gguf_db_dir, user_tags=user_tags, ai_tags=ai_tags, system_tags=system_tags, description=description, keywords=keywords)
            model_data.save_json(replace_existing=replace_existing)
        self.load_models()
    
    def list_available_models(self) -> list[str]:
        print(f"Available models in {self.gguf_db_dir}:")
        models = []
        for model in self.models:
            model:ModelData = model
            if model.name not in models:
                models.append(model.name)
        return models
    
    def list_models_quantizations(self, model_name:str) -> list[str]:
        quantizations = []
        for model in self.models:
            model:ModelData = model
            if model.name == model_name:
                quantizations.append(model.model_quantization)
        return quantizations

    def show_db_info(self) -> None:
        print(f"ModelDB summary:")
        print(f"ModelDB dir: {self.gguf_db_dir}")
        print(f"Number of models: {len(self.models)}")
        print(f"Available models:")
        models_info = {}
        for model in self.models:
            model:ModelData = model
            if model.name not in models_info.keys():
                models_info[model.name] = {}
                models_info[model.name]["quantizations"] = []
                models_info[model.name]["description"] = model.description
                models_info[model.name]["keywords"] = model.keywords
            if model.model_quantization not in models_info[model.name]["quantizations"]:
                models_info[model.name]["quantizations"].append(model.model_quantization)
        
        for model_name, models_info in models_info.items():
            print(f"\t{model_name}:")
            print(f"\t\tQuantizations: {models_info['quantizations']}")
            print(f"\t\tKeywords: {models_info['keywords']}")
            print(f"\t\tDescription: {models_info['description']}")
            print(f"\t-------------------------------")




        


        
        
    