from setuptools import setup

setup(
    name='conformalizer',
    version='0.1.0',    
    description='A package to compute conformal predictions for regression and classification',
    url='https://github.com/carlossisto12/conformalizer',
    author='Carlos Sisto',
    author_email='carlosmiguel.sisto@gmail.com',
    packages=['conformalizer'],
    install_requires=['numpy', 
                      'pandas>=1.0',
                      'scikit-learn>=1.0',
                      'xgboost==1.6.0',
                      'statsmodels~=0.13'
                      ])
