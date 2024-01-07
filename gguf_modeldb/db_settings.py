from util_helper.file_handler import join_paths, get_directory
import os
PACKAGE_DIR = get_directory(__file__)
VERIFIED_MODELS_DB_DIR = join_paths(PACKAGE_DIR, "gguf_models/")
VERIFIED_MODELS_DB_DIR = os.path.abspath(VERIFIED_MODELS_DB_DIR)