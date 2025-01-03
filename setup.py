from setuptools import Command, find_packages, setup

__lib_name__ = "STIFT"
__lib_version__ = "1.0.0"
__description__ = "STIFT: Spatiotemporal Integration Framework for Transcriptomics"
__url__ = "https://github.com/TheLittleJimmy/STIFT"
__author__ = "Ji Qi"
__author_email__ = "qiji@link.cuhk.edu.hk"
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "Deep learning", "Graph attention auto-encoder"]

# Update install_requires using a subset of your requirements:
__requires__ = [
    "anndata==0.10.8",
    "matplotlib==3.9.0",
    "numpy==1.26.0",
    "pandas==2.2.2",
    "requests",
    "scanpy==1.10.3",
    "scikit-learn==1.5.1",
    "scipy==1.12.0",
    "torch==2.3.0+cu121",
    "networkx==3.1",
    "POT==0.9.4",  # Changed from ot==0.9.4
]

setup(
    name=__lib_name__,
    version=__lib_version__,
    description=__description__,
    url=__url__,
    author=__author__,
    author_email=__author_email__,
    license=__license__,
    packages=find_packages(),
    install_requires=__requires__,
    zip_safe=False,
    include_package_data=True)
