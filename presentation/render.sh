#!/bin/bash
pdflatex -interaction=nonstopmode presentation.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode presentation.tex > /dev/null 2>&1
rm -f *.aux *.log *.out *.nav *.snm *.toc *.fls *.fdb_latexmk
mv presentation.pdf 2_cmpe346_final_presentation.pdf
echo "Done → 2_cmpe346_final_presentation.pdf"
