from setuptools import setup, find_packages

setup(
    name="lucaone",
    version="1.1.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=1.13.1",
        "transformers>=4.26.0",
    ],
    author="YongHe",
    description="LucaOne: LucaGPLM Model compatible with HuggingFace Transformers",
)

