from setuptools import setup, find_packages
from pkg_resources import DistributionNotFound, get_distribution


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


install_deps = ['numpy']

if get_dist('tensorflow') is None and get_dist('tensorflow_gpu') is None:
    install_deps.append('tensorflow')

setup(
  name = 'BertLibrary',   
  packages = find_packages(),
  version = '0.0.4',     
  license='MIT',        
  description = 'BaaL is a Tensorflow library for quick and easy training and finetuning of models based on Bert', 
  author = 'KPI6 Research',                   
  author_email = 'info@kpi6.com',    
  url = 'https://github.com/kpi6research/Bert-as-a-Library',   
  download_url = 'https://github.com/kpi6resea