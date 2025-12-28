from setuptools import setup, find_packages

setup(
    name="OAM_KIST",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
    author="Youngjun Kim",
    author_email="kyjun0915@kist.re.kr",z
    description="Quantum information and technology using OAM states and SLM for KIST research",
    python_requires=">=3.7",
)