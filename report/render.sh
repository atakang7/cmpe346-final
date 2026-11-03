#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -halt-on-error -interaction=nonstopmode template346.tex
pdflatex -halt-on-error -interaction=nonstopmode template346.tex
rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk
mv template346.pdf 2_cmpe346_final.pdf
echo "Done → 2_cmpe346_final.pdf"
