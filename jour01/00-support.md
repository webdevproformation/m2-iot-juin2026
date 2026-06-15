# créer un fichier md 
# pandoc convertion md => html


```sh
pandoc \
    src/{sujet_name}.md \
    --template src/template.html \
    --number-sections \
    -s  \
    -o src/{sujet_name}.html
```

# wkhtmltopdf convertion html => pdf 

```sh
wkhtmltopdf \
    --enable-local-file-access \
    --page-size A4 \
    --margin-top 25mm --margin-bottom 20mm --margin-left 15mm --margin-right 15mm \
    --header-html src/header.html \
    --header-spacing 5 \
    --footer-right "Page [page] / [topage]" \
    --footer-font-size 9 \
    --footer-spacing 5 \
    cover src/cover-1.html \
    src/{sujet_name}.html  \
    {sujet_name}.pdf
```

# ghostscript convertion pdf => pdf  (image)

```sh
gs -dSAFER -dBATCH -dNOPAUSE \
    -sDEVICE=pdfwrite \
    -dPDFSETTINGS=/prepress \
    -dNoOutputFonts \
    -sOutputFile={sujet_name}-r.pdf \
    {sujet_name}.pdf
```