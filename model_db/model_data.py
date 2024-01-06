import time
import requests
from typing import Union, Optional

from util_helper.file_handler import *

__all__ = ['ModelData']
"""
Models data format:
{
    "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q2_K.gguf",
    "gguf_file_path": "./models_db/gguf_models\\mistral-7b-instruct-v0.1.Q2_K.gguf",
    "model_name": "mistral-7b-instruct-v0.1",
    "model_quantization": "Q2_K",
    "description": "he Mistral-7B-Instruct-v0.1 Large Language Model (LLM) \n    is a instruct fine-tuned version of the Mistral-7B-v0.1 \n    generative text model using a variety of publicly available conversation datasets.",
    "keywords": [
        "mistral",
        "v0.1",
        "apache",
        "free",
        "uncesored",
        "instruct"
    ],
    "user_tags": {
        "open": "[INST]",
        "close": "[/INST]"
    },
    "ai_tags": {
        "open": "",
        "close": ""
    },
    "system_tags": {
        "open": null,
        "close": null
    },
    "save_dir": "./models_db/gguf_models"
}
"""


class ModelData:
    def __init__(self, 
        gguf_url:str,
        db_dir:str,
        user_tags:Union[dict, list, set] = ("", ""), 
        ai_tags:Union[dict, list, set] = ("", ""),
        system_tags:Optional[Union[dict, list, set]] = None,
        description:Optional[str] = None, 
        keywords:Optional[list] = None,
        ):

        #init all as None
        self.gguf_url = None
        self.gguf_file_path = None
        self.name = None
        self.model_quantization = None
        self.description = None
        self.keywords = None
        self.user_tags = None
        self.ai_tags = None
        self.system_tags = None
        self.save_dir = None

        #set values
        self.gguf_url = gguf_url
        self.set_save_dir(db_dir)
        self.gguf_file_path = self._url_to_file_path(db_dir, gguf_url) 
        self.name = self._url_extract_model_name(gguf_url)
        self.model_quantization = self._url_extract_quantization(gguf_url)
        self.description = description if description is not None else ""
        self.keywords = keywords if keywords is not None else []
        self.set_tags(ai_tags, user_tags, system_tags)

    def __str__(self) -> str:
        t = f"""ModelData(
            ---required---
            gguf_url: {self.gguf_url},
            ---required with defaults--- 
            save_dir: {self.save_dir},
            user_tags: {self.user_tags},
            ai_tags: {self.ai_tags},
            ---optionally provided, no defaults---
            system_tags: {self.system_tags},
            description: {self.description},
            keywords: {self.keywords},
            ---automatically generated---
            gguf_file_path: {self.gguf_file_path},
            model_name: {self.name},
            model_quantization: {self.model_quantization}
        )"""
        return t
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __dict__(self) -> dict:
        return self.to_dict()
    
    @staticmethod
    def _hf_url_to_download_url(url) -> str:
        #to download replace blob with resolve and add download=true
        if not "huggingface.co" in url:
            raise ValueError(f"Invalid url: {url}, must be a huggingface.co url, other sources aren't implemented yet.")
        url = url.replace("blob", "resolve")
        if url.endswith("/"):
            url = url[:-1]
        if not url.endswith("?download=true"):
            url = url + "?download=true"
        return url
    
    @staticmethod    
    def _url_to_file_path(save_dir:str, url:str)->str:
        #create_dirs_for(save_dir)
        file_path = join_paths(save_dir, ModelData._url_extract_file_name(url))
        return file_path 
    
    @staticmethod
    def _url_extract_file_name(url:str) -> str:
        f_name =  url.split("/")[-1]
        if is_file_format(f_name, ".gguf"):
            return f_name
        else:
            raise ValueError(f"File {f_name} is not a gguf file.")
    
    @staticmethod
    def _url_extract_quantization(url:str) -> str:
        quantization = ModelData._url_extract_file_name(url).split(".")[-2]
        return quantization
    
    @staticmethod
    def _url_extract_model_name(url:str) -> str:
        model_name = ModelData._url_extract_file_name(url).split(".")[0:-2]
        return ".".join(model_name)

    def set_ai_tags(self, ai_tags:Union[dict, set[str], list[str], tuple[str]]) -> None:
        if isinstance(ai_tags, dict):
            if "open" in ai_tags and "close" in ai_tags:
                self.ai_tags = ai_tags
            else:
                raise ValueError(f"Invalid user tags: {ai_tags}, for dict tags both 'open' and 'close' keys must be present.")
        elif isinstance(ai_tags, set) or isinstance(ai_tags, list) or isinstance(ai_tags, tuple):
            self.ai_tags = {
                "open": ai_tags[0],
                "close": ai_tags[1]
            }
        else:
            raise TypeError(f"Invalid type for user tags: {type(ai_tags)}, must be dict, set or list.")
        
    def set_user_tags(self, user_tags:Union[dict, set[str], list[str], tuple[str]]) -> None:
        if isinstance(user_tags, dict):
            if "open" in user_tags and "close" in user_tags:
                self.user_tags = user_tags
            else:
                raise ValueError(f"Invalid user tags: {user_tags}, for dict tags both 'open' and 'close' keys must be present.")
        elif isinstance(user_tags, set) or isinstance(user_tags, list) or isinstance(user_tags, tuple):
            self.user_tags = {
                "open": user_tags[0],
                "close": user_tags[1]
            }
        else:
            raise TypeError(f"Invalid type for user tags: {type(user_tags)}, must be dict, set or list.")

    def set_system_tags(self, system_tags:Union[dict, set[str], list[str], tuple[str]]) -> None:
        if isinstance(system_tags, dict):
            if "open" in system_tags and "close" in system_tags:
                self.system_tags = system_tags
            else:
                raise ValueError(f"Invalid system tags: {system_tags}, for dict tags both 'open' and 'close' keys must be present.")
        elif isinstance(system_tags, set) or isinstance(system_tags, list) or isinstance(system_tags, tuple):
            self.system_tags = {
                "open": system_tags[0],
                "close": system_tags[1]
            }
        else:
            raise TypeError(f"Invalid type for system tags: {type(system_tags)}, must be dict, set or list.")
    def set_tags(self, 
                 ai_tags:Optional[Union[dict, set[str], list[str], tuple[str]]],
                 user_tags:Optional[Union[dict, set[str], list[str], tuple[str]]],
                 system_tags:Optional[Union[dict, set[str], list[str], tuple[str]]],
                 ) -> None:
        if ai_tags is not None:
            self.set_ai_tags(ai_tags)
        if user_tags is not None:
            self.set_user_tags(user_tags)
        if system_tags is not None:
            self.set_system_tags(system_tags)

    def set_save_dir(self, save_dir:str) -> None:
        self.save_dir = save_dir
        self.gguf_file_path = self._url_to_file_path(save_dir, self.gguf_url)

    def to_dict(self):
        model_data = {
            "url": self.gguf_url,
            "gguf_file_path": self.gguf_file_path,
            "model_name": self.name,
            "model_quantization": self.model_quantization, 
            "description": self.description,
            "keywords": self.keywords,
            "user_tags": self.user_tags,
            "ai_tags": self.ai_tags,
            "system_tags": self.system_tags,
            "save_dir": self.save_dir,
        }
        return model_data
    @staticmethod
    def from_dict(model_data:dict) -> "ModelData":
        url = model_data["url"]
        save_dir = model_data["save_dir"]
        description = model_data["description"] if "description" in model_data else None
        keywords = model_data["keywords"] if "keywords" in model_data else None
        user_tags = model_data["user_tags"]
        ai_tags = model_data["ai_tags"]
        system_tags = model_data["system_tags"] if "system_tags" in model_data else None
        new_model_data = ModelData(url, save_dir, user_tags, ai_tags, system_tags, description, keywords)
        return new_model_data

    def is_downloaded(self) -> bool:
        """Checks if the model file is downloaded."""
        return does_file_exist(self.gguf_file_path)
    
    def has_json(self) -> bool:
        """Checks if the model is in the db."""
        return does_file_exist(self.json_path())
    
    def download_gguf(self, force_redownload:bool=False) -> str:
        print(f"Preparing {self.gguf_file_path}\n for {self.name} : {self.model_quantization}...")
        if not does_file_exist(self.gguf_file_path) or force_redownload:
            print(f"Downloading {self.name} : {self.model_quantization}...")
            gguf_download_url = self._hf_url_to_download_url(self.gguf_url)
            response = requests.get(gguf_download_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024000  # 100 KB
            progress_bar = f"Please wait, downloading {self.name} : {self.model_quantization}: {{0:0.2f}}% | {{1:0.3f}}/{{2:0.3f}} GB) | {{3:0.3f}} MB/s"
            unfinished_save_path = self.gguf_file_path + ".unfinished"
            with open(unfinished_save_path, "wb") as f:
                downloaded_size = 0
                start_time = time.time()
                elapsed_time = 0
                downloaded_since_last = 0
                for data in response.iter_content(block_size):
                    downloaded_size += len(data)
                    downloaded_since_last += len(data)
                    f.write(data)
                    elapsed_time = time.time() - start_time
                    download_speed = (downloaded_since_last*10/(1024**3)) / elapsed_time if elapsed_time > 0 else 0
                    progress = downloaded_size / total_size * 100
                    gb_downloaded = downloaded_size/(1024**3)
                    gb_total = total_size/(1024**3)
                    if elapsed_time >= 1:
                        print(progress_bar.format(progress, gb_downloaded, gb_total, download_speed), end='\r')
                        downloaded_since_last = 0
                        start_time = time.time()
            print(progress_bar.format(100, gb_downloaded, gb_total, download_speed))
            rename_file(unfinished_save_path, self.gguf_file_path)
        else:
            print(f"File {self.gguf_file_path} already exists. Skipping download.")
        return self.gguf_file_path
    
    def json_path(self) -> str:
        return change_extension(self.gguf_file_path, ".json")
    
    def save_json(self, replace_existing:bool=True) -> str:
        if replace_existing or not self.has_json():
            save_json_file(self.json_path(), self.to_dict())
        else:
            print(f"File {self.json_path()} already exists and replace_existing={replace_existing}. Skipping save.")
        return self.json_path()
    
    @staticmethod
    def from_json(json_file_path:str) -> "ModelData":
        model_data = load_json_file(json_file_path)
        return ModelData.from_dict(model_data)

    @staticmethod
    def from_url(url:str, save_dir:str, user_tags:Union[dict, list, set] = ("", ""), ai_tags:Union[dict, list, set] = ("", ""), system_tags:Union[dict, list, set] = ("", ""),
                  description:Optional[str] = None, keywords:Optional[list] = None) -> "ModelData":
        return ModelData(url, save_dir, user_tags, ai_tags, system_tags, description, keywords)

    def model_path(self) -> str:
        return self.gguf_file_path

    @staticmethod
    def from_file(gguf_file_path:str, save_dir:Optional[str]=None, user_tags:Union[dict, list, set] = ("", ""), ai_tags:Union[dict, list, set] = ("", ""), system_tags:Union[dict, list, set] = ("", ""),
                  description:Optional[str] = None, keywords:Optional[list] = None) -> "ModelData":
        #creates a model where url is also the file path
        save_dir = get_directory(gguf_file_path) if save_dir is None else save_dir
        url = gguf_file_path
        return ModelData(url, save_dir, user_tags, ai_tags, system_tags, description, keywords)

    def get_ai_tag_open(self) -> str:
        return self.ai_tags["open"]
    
    def get_ai_tag_close(self) -> str:
        return self.ai_tags["close"]
    
    def get_user_tag_open(self) -> str:
        return self.user_tags["open"]
    
    def get_user_tag_close(self) -> str:
        return self.user_tags["close"]
    
    def get_system_tag_open(self) -> str:
        return self.system_tags["open"]
    
    def get_system_tag_close(self) -> str:
        return self.system_tags["close"]
    
    def get_ai_tags(self) -> list[str]:
        return [self.get_ai_tag_open(), self.get_ai_tag_close()]
    
    def get_user_tags(self) -> list[str]:
        return [self.get_user_tag_open(), self.get_user_tag_close()]
    
    def get_system_tags(self) -> list[str]:
        return [self.get_system_tag_open(), self.get_system_tag_close()]
    
    def has_system_tags(self) -> bool:
        return self.system_tags is not None