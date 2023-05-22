from setuptools import find_packages, setup

setup(
    name="trident-core",
    version="0.1.0",
    description="A declarative deep learning framework",
    license="Apache",
    packages=find_packages(exclude=["tests", "tests/*", "src", "src/*"]),
    install_requires=[
        "pytorch-lightning",
        "hydra-core",
        "hydra-colorlog",
        "hydra-optuna-sweeper",
        "numpy",
        "python-dotenv",
        "rich",
        "scikit-learn",
        "seaborn",
    ],
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
