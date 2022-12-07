import setuptools

setuptools.setup(
    name="rl_testing",
    version="0.1dev",
    description="",
    # long_description=open("README.md").read(),
    url="https://github.com/luk-s/rl-testing-experiments",
    install_requires=[
        "numpy==1.23.4",
        "asyncssh==2.12.0",
        "python-chess==1.999",
        "imgkit==1.2.2",
        "matplotlib==3.6.1",
        "pandas==1.5.1",
        "scipy==1.9.3",
        "networkx==2.8.8",
        "netwulf==0.1.5",
    ],
    author="Lukas Fluri",
    author_email="lukas.fluri@protonmail.ch",
    license="MIT",
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
)
