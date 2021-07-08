from setuptools import find_packages, setup

setup(
    name='stochastiCV',
    packages=find_packages(
        where = 'stochastiCV'
    ),
    version='0.1.5',
    description='A method of cross-validation based on scikit-learn that splits data into train/valid/test splits two or more times (using random or assigned seed values) and then repeats the model multiple times using different seeds. This function enables a more statistical and scientific method of investigating model performance.',
    author='Michael J Kleiman',
    author_email='michael@kleiman.me',
    license='BSD-3',
)
