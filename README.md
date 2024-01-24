# SIAG/FME Code Quest 2023
This repository contains the code used for the SIAG/FME Code Quest 2023.

# Usage
The code is intended to run on Google Colab without any kind of supplemental code installation.

## Note

**Please note:** The code relies on the compiled cython files amm_cython.c and amm_cython.cpython-39-x86_64-linux-gnu.so.
                 These files can no longer work in case of a Cython module upgrade in Colab. In such case, it it better to
                 recompile them on Google Colab using
                 ```bash
                !python setup.py build_ext --inplace

Simply run the main.py file
'''bash
!python main.py
   

