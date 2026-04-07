from setuptools import find_packages, setup


def read_requirements(path: str) -> list[str]:
    requirements = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                requirements.append(line)
    return requirements


setup(
    name="slime",
    version="0.2.2",
    author="slime Team",
    python_requires=">=3.10",
    packages=find_packages(include=["slime", "slime.*", "slime_plugins", "slime_plugins.*"]),
    include_package_data=True,
    install_requires=read_requirements("requirements.txt"),
    extras_require={"fsdp": ["torch>=2.0"]},
)
