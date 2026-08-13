import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))

autodoc_mock_imports = [
    # GUI
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",

    "numpy",
    "scipy",
    "matplotlib",

    "serial",        
    "pyvisa",
    "zaber_motion",
    "thorlabs_apt",
    "ThorlabsPM100",
    "pylablib",

    "yaml",
]

autodoc_mock_imports += [
    "PyQt5.sip",
    "sip",
    "pylablib.devices",
    "pylablib.core",
]


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dlabcontroller'
copyright = '2026, Melvin Redon'
author = 'Melvin Redon'
release = '2.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]
extensions += [
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
]
autosectionlabel_prefix_document = True


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_static_path = ['_static']
