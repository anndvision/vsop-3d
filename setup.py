from setuptools import setup, find_packages

setup(
    name="vsop-3d",
    version="0.0.0",
    description="",
    url="https://github.com/anndvision/vsop-3d",
    author="Andrew Jesson and Yiding Jiang",
    author_email="andrew.d.jesson@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "gym==0.23.1",
        "six==1.16.0",
        "wandb==0.15.8",
        "pandas==2.0.3",
        "procgen==0.10.7",
        "seaborn==0.12.2",
        "ray[tune]==2.5.1",
        "shortuuid==1.0.11",
        "hpbandster==0.7.4",
        "matplotlib==3.7.2",
        "configspace==0.7.1",
        "tensorboard==2.14.0",
        "bayesian-optimization==1.4.3",
        "fsspec",
    ],
)
