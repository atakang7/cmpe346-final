#!/bin/bash
pdflatex -interaction=nonstopmode template346.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode template346.tex > /dev/null 2>&1
rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk
mv template346.pdf 2_cmpe346_final.pdf
echo "Done → 2_cmpe346_final.pdf"
