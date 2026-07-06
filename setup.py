from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="noirag",
    version="1.0.1",
    description="Noise-Aware Retrieval-Augmented Generation preprocessing engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sourav Roy & Team NoiRAG",
    packages=find_packages(include=["noirag", "noirag.*"]),
    install_requires=[
        "symspellpy>=6.7",
        "regex>=2023.0",
        "requests>=2.28",
        "python-dotenv>=1.0",
        "pandas>=2.0",
        "pyarrow>=12.0",
        "scipy>=1.10.0",
        "codecarbon>=2.3.0",
    ],
    python_requires=">=3.8",
)
