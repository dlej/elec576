# ELEC 576 – Introduction to Deep Learning (F17)

This project contains code, notebooks, etc, developed by me for ELEC 576
in the Fall 2017 semester at Rice University. Feel free to use anything
you find here.

## Generating PDFs with `nbconvert`

To make the pdf conversion a bit nicer with `nbconvert`, use the
template and class files (`dlejeune_nb.tplx` and `dlejeune_hw.cls`,
respectively) to do the conversion. `dlejeune_hw.cls` will need to be
installed to an appropriate directory (e.g., `~/texmf/tex/latex/misc`),
then you can convert to pdf with

```
jupyter nbconvert --to pdf --template ../dlejeune_nb.tplx <<filename>>.ipynb
```