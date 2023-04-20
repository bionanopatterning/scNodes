from setuptools import setup, find_packages

# how to release:
# update version number in config manually

# push to pypi:
# python setup.py sdist
# twine upload dist/*
#
# bundle as .exe:
# pyinstaller scNodes.spec --noconfirm
# find __main__ folder in PycharmProjects/.../scNodes/dist, .zip it, upload to GitHub.

setup(
    name='scNodes',
    version='1.1.14',
    packages=find_packages(),
    url='https://github.com/bionanopatterning/scNodes',
    license='GPL v3',
    author='mgflast',
    author_email='m.g.f.last@lumc.nl',
    description='Correlating super-resolution fluorescence and cryoEM\ngithub.com/bionanopatterning/scNodes',
    package_data={'': ['*.png', '*.glsl', '*.whl', '*.tar.gz', '*.pdf', '*.scn']},
    include_package_data=False,  # weirdly, the above filetypes _are_ included when this parameter is set to False.
    install_requires=[
        "colorcet>=3.0.0",
        "dill>=0.3.5.1",
        "glfw>=2.5.5",
        "joblib>=1.1.0",
        "matplotlib>=3.5.3",
        "mrcfile>=1.4.3",
        "numpy>=1.3.0",
        "opencv-python>=4.6.0.66",
        "pandas>=1.3.5",
        "Pillow>=9.2.0",
        "psutil>=5.9.2",
        "PyOpenGl>=3.1.6",
        "PyWavelets>=1.3.0",
        "pyperclip>=1.8.2",
        "pystackreg>=0.2.6.post1",
        "pyimgui-wheels",
        "scikit-image>=0.19.3",
        "tifffile>=2021.11.2" ## was 2022.8.12
    ]
)
