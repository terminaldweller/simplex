from setuptools import setup


setup(
    name="dsimplex",
    version="0.1.11",
    description="solves LP problems using the simplex method",
    url="https://github.com/terminaldweller/simplex",
    author="terminaldweller",
    author_email="thabogre@gmail.com",
    license="GPL3",
    packages=["dsimplex"],
    zip_safe=False,
    entry_points={"console_scripts": ["dsimplex = dsimplex.simplex:dsimplex"]},
)
