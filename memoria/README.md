# Modelo de memoria de TFG (Grao en Enxeñaría Informática da FIC UDC)

Este proxecto LaTeX constitúe un modelo de referencia para as memorias de Traballo Fin de Grao
do Grao en Enxeñaría Informática da Facultade de Informática da Universidade da Coruña.

## Estrutura

  1) Ficheiros de autoría, contribucións, licenza e atribución

     > `AUTHOR`
     >
     > `CONTRIBUTING.md`
     >
     > `COPYING`
     >
     > `CREDITS`

  2) Ficheiro de estilo: `estilo_tfg.sty`

  3) Ficheiro principal: `memoria_tfg.tex`

  4) Directorios:

     > `anexos/`		Contén os capítulos con materiais adicionais.
     >
     > `bibliografia/`	Contén a bibliografía e outros posibles índices (termos, glosario).
     >
     > `contido/`		Contén os capítulos da memoria.
     >
     > `imaxes/`		Contén as imaxes da memoria.
     >
     > `portada/`		Contén a portada, resumo e palabras chave.

## Xeración da versión PDF

A versión PDF pódese xerar empregando a ferramenta `latexmk`, que asegura o correcto procesamento
de índices, bibliografía e referencias:

     latexmk -xelatex memoria_tfg.tex

A ferramenta `latexmk` pódese empregar de xeito que monitorice o proxecto e recompile automaticamente
a memoria en caso de producirse cambios nos diferentes ficheiros que a conforman:

     latexmk -xelatex -pvc memoria_tfg.tex

## Eliminación dos ficheiros auxiliares

Os ficheiros auxiliares xerados por `pdflatex` ou `latexmk` poden eliminarse doadamente con:

     latexmk -c
