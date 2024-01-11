import bs4
import requests
from typing import Union, Optional

from util_helper.file_handler import create_dir, list_files_in_dir, copy_large_file, get_absolute_path
from util_helper.compare_strings import compare_two_strings
from .model_data import ModelData
from .db_settings import VERIFIED_MODELS_DB_DIR

class ModelDB:
    """Class for managing a database of ModelData objects.

    Handles loading models from a directory, searching, adding new models,
    and interfacing with HuggingFace to import models.

    Attributes:
        gguf_db_dir (str): Path to directory containing ModelData files
        models (List[ModelData]): List of ModelData objects  
    """
    
    def __init__(self, model_db_dir:Optional[str]=None, copy_verified_models=True):
        """Initialize ModelDB object.

        Args:
            model_db_dir (str, optional): Path to database directory. Defaults to VERIFIED_MODELS_DB_DIR.
            copy_verified_models (bool, optional): Whether to copy example models to the new directory. Defaults to True.
        """
        self.gguf_db_dir = None
        self.models = []

        if model_db_dir is None:
            model_db_dir = VERIFIED_MODELS_DB_DIR
        else:
            model_db_dir = get_absolute_path(model_db_dir)
        self.set_model_db_dir(model_db_dir)


        if model_db_dir != VERIFIED_MODELS_DB_DIR:
            if copy_verified_models:
                print(f"Copying examples to {model_db_dir}...")
                for file in list_files_in_dir(VERIFIED_MODELS_DB_DIR, False, True, [".json"], absolute=True):
                    f_mdt = ModelData.from_json(file)
                    f_mdt.set_save_dir(model_db_dir)
                    f_mdt.save_json()
                    print(f"Saved a copy of {file} to {model_db_dir}.")
        else:
            print(f"Using default model db dir: {model_db_dir}, reconfiguring models...")
            for file in list_files_in_dir(VERIFIED_MODELS_DB_DIR, False, True, [".json"], absolute=True):
                f_mdt = ModelData.from_json(file)
                f_mdt.set_save_dir(model_db_dir)
                f_mdt.save_json()
                print(f"Reconfigured {file}.")

        self.load_models()
    
    def set_model_db_dir(self, model_db_dir:str) -> None:
        """Set the database directory.

        Args:
            model_db_dir (str): Path to database directory
        """
        print(f"ModelDB dir set to {model_db_dir}.")
        self.gguf_db_dir = create_dir(model_db_dir)
    
    def load_models(self) -> None:
        """Load ModelData objects from the database directory."""
        self.models = []
        files = list_files_in_dir(self.gguf_db_dir, False, True, [".json"], absolute=True)
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
                   treshold:float=0.6,
                   only_downloaded:bool=False) -> Union[None, list]:
        """Search for models based on name, quantization, and keywords.

        Args:
            name_query (str, optional): Search query for name
            quantization_query (str, optional): Search query for quantization 
            keywords_query (str, optional): Search query for keywords
            treshold (float, optional): Minimum similarity score threshold. Defaults to 0.6.

        Returns:
            Union[None, list]: Sorted list of models exceeding threshold,
                               or None if no query provided
        """
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
                    scoring_models_dict[id]["score"] += top_name_score*4
                #print(f"Model {model_name} {model_quantization} top score: {top_name_score} treshold: {treshold}")
            if quantization_query is not None:
                #print(f"Searching for quantization: {quantization_query}")
                quantization_score = compare_two_strings(quantization_query, model_quantization)
                if quantization_score > treshold:
                    scoring_models_dict[id]["score"] += quantization_score*2
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
        if only_downloaded:
            sorted_models = [x for x in sorted_models if x.is_downloaded()]
        #print(f"Found {len(sorted_models)} models.")
        #print(sorted_models)
        return sorted_models
    
    def find_model(self, name_query:Optional[str]=None, 
                   quantization_query:Optional[str]=None, 
                   keywords_query:Optional[str]=None,
                   only_downloaded:bool=False
                   ) -> Optional[ModelData]:
        """Find top matching model based on queries.

        Args:
            name_query (str, optional): Search query for name
            quantization_query (str, optional): Search query for quantization
            keywords_query (str, optional): Search query for keywords

        Returns:
            Optional[ModelData]: Top matching ModelData object or None
        """
        sorted_models = self.find_models(name_query, quantization_query, keywords_query, only_downloaded=only_downloaded)
        if sorted_models is None or len(sorted_models) == 0:
            if len(self.models) == 0:
                print(f"There were no models to be searched. Try importing a verified model or using the defualt db dir.")
            raise Exception(f"Could not find a model matching the query: {name_query} {quantization_query} {keywords_query}")
        else:
            #print(f"Found {len(sorted_models)} models.")
            #print(sorted_models)
            return sorted_models[0]
        
    def get_model_by_url(self, url:str) -> Optional[ModelData]:
        """Get ModelData by exact URL match.

        Args:
            url (str): ggUF URL

        Returns:
            Optional[ModelData]: Matching ModelData or None if not found
        """
        for model in self.models:
            model:ModelData = model
            if model.gguf_url == url:
                return model
        return None
    
    def get_model_by_gguf_path(self, gguf_path:str) -> Optional[ModelData]:
        """Get ModelData by exact ggUF path match.

        Args:
            gguf_path (str): ggUF path

        Returns:
            Optional[ModelData]: Matching ModelData or None if not found
        """
        for model in self.models:
            model:ModelData = model
            if model.gguf_file_path == gguf_path:
                return model
        return None
        
    def add_model_data(self, model_data:ModelData, save_model=True) -> None:
        """Add a ModelData object to the database.

        Args:
            model_data (ModelData): ModelData object to add
            save_model (bool, optional): Whether to save ModelData to file. Defaults to True.
        """
        self.models.append(model_data)
        if save_model:
            model_data.save_json()
    
    def add_model_by_url(self, url:str, ) -> None:
        """Add a model by URL.

        Args:
            url (str): ggUF URL
        """
        model_data = ModelData(url, db_dir=self.gguf_db_dir)
        self.add_model_data(model_data)

    def add_model_by_json(self, json_file_path:str) -> None:
        """Add a model from a JSON file.

        Args:
            json_file_path (str): Path to ModelData JSON file
        """
        model_data = ModelData.from_json(json_file_path)
        self.add_model_data(model_data)

    def save_all_models(self) -> None:
        """Save all ModelData objects to file."""
        for model in self.models:
            model:ModelData = model
            model.save_json()
                
    @staticmethod
    def _model_links_from_repo(hf_repo_url:str):
        """Extract ggUF model links from a HuggingFace repo page.

        Args:
            hf_repo_url (str): URL of HuggingFace model repo

        Returns:
            list: List of ggUF URLs
        """
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
        """Load model data from a HuggingFace repo page.

        Args:
            hf_repo_url (str): URL of HuggingFace model repo
            user_tags (Optional[list[str]], optional): User tags to apply. Defaults to None.
            ai_tags (Optional[list[str]], optional): AI tags to apply. Defaults to None.
            system_tags (Optional[list[str]], optional): System tags to apply. Defaults to None.
            keywords (Optional[list[str]], optional): Keywords to apply. Defaults to None.
            description (Optional[str], optional): Description to apply. Defaults to None.

        Returns:
            list: List of loaded ModelData objects
        """
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
        """Import models from a HuggingFace repo page.

        Args:
            hf_repo_url (str): URL of HuggingFace model repo  
            user_tags (Optional[list[str]], optional): User tags to apply. Defaults to None.
            ai_tags (Optional[list[str]], optional): AI tags to apply. Defaults to None.
            system_tags (Optional[list[str]], optional): System tags to apply. Defaults to None.
            keywords (Optional[list[str]], optional): Keywords to apply. Defaults to None.
            description (Optional[str], optional): Description to apply. Defaults to None.
            replace_existing (bool, optional): Whether to overwrite existing files. Defaults to False.
        """
        #create model data from hf repo
        model_links = ModelDB._model_links_from_repo(hf_repo_url)
        print(f"Loaded {len(model_links)} models from {hf_repo_url}.")
        for model_link in model_links:
            model_data = ModelData(gguf_url=model_link, db_dir=self.gguf_db_dir, user_tags=user_tags, ai_tags=ai_tags, system_tags=system_tags, description=description, keywords=keywords)
            model_data.save_json(replace_existing=replace_existing)
        self.load_models()
    

    def import_verified_model(self, 
                              name_search:Optional[str]=None,
                              quantization_search:Optional[str]=None,
                              keywords_search:Optional[str]=None,
                              copy_gguf:bool=True) -> None:
        """Import a verified model from the verified model database with ready configurations into your selected db dir.
        Use this to selectively add models from the verified model database to your own database.
        Models inlcude official dolphin, mistral, mixtral, solar and zephyr models in all available quantizations. 
        Args:
            name_search (Optional[str], optional): Search query for name. Defaults to None.
            quantization_search (Optional[str], optional): Search query for quantization. Defaults to None.
            keywords_search (Optional[str], optional): Search query for keywords. Defaults to None.
        """
        if self.gguf_db_dir == VERIFIED_MODELS_DB_DIR:
            print("Cannot import verified model to the default database directory. All models should be already available here.")
        else:
            vmdb = ModelDB()
            if name_search is None and quantization_search is None and keywords_search is None:
                print(f"Importing all verified models to {self.gguf_db_dir}...")
                models = vmdb.models
            else:
                print(f"Importing a verified model matching {name_search} {quantization_search} {keywords_search} to {self.gguf_db_dir}...")
                models = [vmdb.find_model(name_search, quantization_search, keywords_search)]
            for model in models:
                if copy_gguf and model.is_downloaded():
                    source_file = model.gguf_file_path
                    target_file = model.gguf_file_path.replace(vmdb.gguf_db_dir, self.gguf_db_dir)
                    print(f"Copying {source_file} to {target_file}...")
                    copy_large_file(source_file, target_file)
                model.set_save_dir(self.gguf_db_dir)
                model.save_json()
            self.load_models()

    def list_available_models(self) -> list[str]:
        """Get a list of available model names.

        Returns:
            list[str]: List of model names
        """
        print(f"Available models in {self.gguf_db_dir}:")
        models = []
        for model in self.models:
            model:ModelData = model
            if model.name not in models:
                models.append(model.name)
        return models
    
    def list_models_quantizations(self, model_name:str) -> list[str]:
        """Get list of quantizations for a model.

        Args:
            model_name (str): Name of model

        Returns:
            list[str]: List of quantizations
        """
        quantizations = []
        for model in self.models:
            model:ModelData = model
            if model.name == model_name:
                quantizations.append(model.model_quantization)
        return quantizations

    def show_db_info(self) -> None:
        """Print summary information about the database."""
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