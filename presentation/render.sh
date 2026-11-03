#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -halt-on-error -interaction=nonstopmode presentation.tex
pdflatex -halt-on-error -interaction=nonstopmode presentation.tex
rm -f *.aux *.log *.out *.nav *.snm *.toc *.fls *.fdb_latexmk
mv presentation.pdf 2_cmpe346_final_presentation.pdf
echo "Done → 2_cmpe346_final_presentation.pdf"
