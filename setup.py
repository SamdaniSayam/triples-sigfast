from setuptools import find_packages, setup

# Read runtime dependencies from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    required = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read long description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="triples-sigfast",
    version="1.1.0",
    author="TripleS Studio",
    description=(
        "An enterprise-grade, JIT-compiled time-series and nuclear physics "
        "analysis engine stress-tested on 100M+ row datasets."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamdaniSayam/triples-sigfast",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
    ],
)
