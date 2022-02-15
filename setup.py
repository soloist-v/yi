from setuptools import find_packages, setup
import sys
import shutil


def get_install_requires():
    install_requires = ["numpy"]
    return install_requires


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append("install")
    setup(
        name='yi',
        version="1.0",
        description='yi autograd',
        long_description="工具接口",
        keywords='yi',
        packages=find_packages(),
        package_data={'yi': ['*.*']},
        python_requires=">=3.7",
        install_requires=get_install_requires(),
        zip_safe=False)
    if sys.argv[1] != "bdist_wheel":
        shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("yi.egg-info", ignore_errors=True)
