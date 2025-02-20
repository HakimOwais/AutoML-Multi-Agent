import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
__version__ = "0.0.0"

REPO_NAME = "AutoML-Multi-Agent"
AUTHOR_USERNAME = "HakimOwais"
SRC_REPO = "source"
AUTHOR_EMAIL = "owaisibnmushtaq@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USERNAME,
    author_email=AUTHOR_EMAIL,
    description="A Multi-Agent LLM framework for full AutoML pipeline ",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/HakimOwais/AutoML-Multi-Agent",
    package_dir={"":"source"},
    packages=setuptools.find_packages(where="source")
)
