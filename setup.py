import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="risky_decisions",  # Replace with your own username
    version="0.0.1",
    author="bkissler",
    author_email="bekipip@gmail.com",
    description="An implementation of major theories of decision under risk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
