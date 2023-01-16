from setuptools import setup

setup(
    name='scNodes',
    version='1.0.0',
    packages=['src', 'src.nodes', 'src.nodes.pysofi', 'src.ceplugins'],
    url='https://github.com/bionanopatterning/scNodes',
    license='GNU GPL v3',
    author='mgflast',
    author_email='m.g.f.last@lumc.nl',
    description='Correlating super-resolution fluroescence and transmission electron cryomicroscopy'
)
