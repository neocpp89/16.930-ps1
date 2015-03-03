TEXSRC = report.tex
PDF = $(TEXSRC:.tex=.pdf)

.PHONY: clean

all: $(PDF)

clean:
	latexmk -C $(TEXSRC);

$(PDF): $(TEXSRC)
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $(TEXSRC); latexmk -c $(TEXSRC);

