from setuptools import setup, find_packages

with open("./README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="gguf_modeldb",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'requests>=2.31.0',
        'beautifulsoup4>=4.9.3',
        'util_helper==0.0.4'
    ],
    package_data={'gguf_modeldb': ['gguf_models/*.json']},
    include_package_data=True,
    author="Łael Al-Halawani",
    author_email="laelhalawani@gmail.com",
    description="A Llama2 quantized gguf model db with over 80 preconfigured models downloadable in one line, easly add your own models or adjust settings. Don't struggle with manual downloads again.",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.1",
        "Topic :: Text Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['llama', 'ai', 'artificial intelligence', 'natural language processing', 'nlp', 'quantization', 'cpu', 'deployment', 'database', 'model', 'models', 'model database', 'model repo', 'model repository', 'model library', 'model libraries',
              'gguf', 'llm'],
    url="https://github.com/laelhalawani/gguf_modeldb",
)