# Building docs

We use Sphinx for generating the API and reference documentation.


## Instructions

After installing NetworkX and its dependencies, install the Python
packages needed to build the documentation by entering

    pip install sphinx
    pip install sphinx-rtd-theme


To build the HTML documentation, enter

    make html

in the ``doc/`` directory.  This will generate a ``build/html`` subdirectory
containing the built documentation.

To completely rebuild the documentation run

    make clean && make html
