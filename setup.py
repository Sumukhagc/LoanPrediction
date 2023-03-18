from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    #This will return the list of all requirements
    requirement=[]
    HYPHEN_E_DOT='-e.'
    with open(file_path) as file_obj:
        requirement=file_obj.readlines()
        requirement=[req.replace('\n','') for req in requirement]
    if HYPHEN_E_DOT in requirement:
        requirement.remove(HYPHEN_E_DOT)

    return requirement

setup(
    name='LoanPrediction',
    version='0.0.1',
    author='Sumukha',
    author_email='sumukhagc@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)