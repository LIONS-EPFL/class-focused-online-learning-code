from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    install_reqs = [x for x in f.readlines() if x[0] != "#"]
setup(
    name="cfol",
    version="0.0",
    packages=["cfol", "cfol.cvar"],
    url="",
    license="",
    author="Thomas Pethick",
    author_email="",
    description="",
    install_requires=install_reqs,
)
