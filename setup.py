from setuptools import setup, find_packages

setup(
    name="microgpt",
    version="0.1.0",
    author="Zain Sharief",
    author_email="zain.sharief21@gmail.com",
    long_description=open("README.md").read(),
    url="https://github.com/ZainSharief/microgpt", 
    license="MIT",
    packages=find_packages(exclude=["tests", "weights"]),
    include_package_data=True,
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
)
