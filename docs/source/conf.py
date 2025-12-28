# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OAM_KIST'
copyright = '2025, Youngjun Kim'
author = 'Youngjun Kim'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # docstring을 문서로 가져오는 기능
    'sphinx.ext.napoleon',     # Google Style/NumPy Style을 해석하는 기능
    'sphinx.ext.viewcode',     # 문서에 [소스 코드 보기] 링크 추가
    'myst_parser',             # Markdown(.md) 파일 지원
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
