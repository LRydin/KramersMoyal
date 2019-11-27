import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kramersmoyal",
    version="0.4",
    author="Leonardo Rydin Gorjao and Francisco Meirinhos",
    author_email="leonardo.rydin@gmail.com",
    description="Calculate Kramers-Moyal coefficients for stochastic process of any dimension, up to any order.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LRydin/KramersMoyal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)
