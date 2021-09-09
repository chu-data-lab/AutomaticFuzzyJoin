import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autofj",
    version="0.0.6",
    author="Peng Li",
    author_email="lipengpublic@gmail.com",
    description="Auto-Program Fuzzy Similarity Joins Without Labeled Examples",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chu-data-lab/AutomaticFuzzyJoin",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        'numpy',
        'pandas',
        'nltk',
        'ngram',
        'editdistance',
        'jellyfish',
        'spacy',
    ],
    include_package_data=True
)