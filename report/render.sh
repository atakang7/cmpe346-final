#!/bin/bash
pdflatex -interaction=nonstopmode template346.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode template346.tex > /dev/null 2>&1
rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk
echo "Done → template346.pdf"
