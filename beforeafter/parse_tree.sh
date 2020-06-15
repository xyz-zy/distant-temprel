#!/bin/bash

cd ${1}

for filename in ${2}; do
  name=${filename##*/}
  base=${name%.txt}
  echo $base
  java -mx2000m edu/stanford/nlp/parser/lexparser/LexicalizedParser -outputFormat "penn" -maxLength 100 edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $filename > ${3}${base}.tree
done

