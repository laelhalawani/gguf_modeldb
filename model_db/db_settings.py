from util_helper.file_handler import join_paths, get_directory
import os

PACKAGE_DIR = get_directory(__file__)
MODEL_EXAMPLES_DB_DIR = join_paths(PACKAGE_DIR, "gguf_models")
MODEL_EXAMPLES_DB_DIR = os.path.abspath(MODEL_EXAMPLES_DB_DIR)
DEFAULT_LOCAL_GGUF_DIR = join_paths("./", "gguf_db")