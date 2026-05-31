#!/bin/bash
pdflatex -interaction=nonstopmode presentation.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode presentation.tex > /dev/null 2>&1
rm -f *.aux *.log *.out *.nav *.snm *.toc *.fls *.fdb_latexmk
echo "Done → presentation.pdf"
