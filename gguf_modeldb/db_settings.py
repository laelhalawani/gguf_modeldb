from util_helper.file_handler import join_paths, get_directory

PACKAGE_DIR = get_directory(__file__)
VERIFIED_MODELS_DB = join_paths(PACKAGE_DIR, "gguf_models/")
#VERIFIED_MODELS_DB = os.path.abspath(VERIFIED_MODELS_DB)
DEFAULT_LOCAL_GGUF_DIR = join_paths("./", "gguf_models/")