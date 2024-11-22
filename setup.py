from setuptools import setup, find_packages

# Read the contents of README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gradient_boosting_model_selection",
    version="1.0.0",
    author="Shubham Dhanavade",
    author_email="sdhanavade@hawk.iit.edu.com",
    description="Implementation of Gradient Boosting and Model Selection (K-Fold CV, Bootstrapping)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mishrasharanya/CS584_Project2",  # Replace with your GitHub URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "run_gradient_boosting=main_gradientboosting:main",
            "run_model_selection=main_modelselection:main",
        ]
    },
)
