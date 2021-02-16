import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="manifoldy", # Replace with your own username
    version="0.3.3",
    author="Raul Lara-Cabrera",
    author_email="lara.cabrera@gmail.com",
    description="A package to generate customizable manifold synthetic datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laracabrera/manifoldy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.4',
    install_requires=[
        'numpy>=1.18',
        'scipy>=1.4'
    ]
)