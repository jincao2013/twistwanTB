from setuptools import setup
from twistwanTB import __version__

packages = [
    'twistwanTB',
    'twistwanTB.wanpy',
    'twistwanTB.core',
]

setup(
    name='twistwanTB',
    version=__version__,
    packages=packages,
    url='https://github.com/jincao2013/twistwanTB',
    license='GPL v3',
    author='jincao',
    author_email='caojin.phy@gmail.com',
    description='Wannier TB model for TMG',
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'h5py', 'matplotlib'],
)
