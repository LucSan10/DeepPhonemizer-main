#!/bin/bash
arr=( $(uchardet ./lexicons/*) )
for (( i=0; i<=${#arr[@]}-1; i+=2))
do
	file="${arr[$i]##*/}"
	file="${file%:*}"
	enc="${arr[$i+1]}"

	echo ${file}

	$(iconv -f "${enc}" -t UTF-8 ./lexicons/"${file}" -o ./lexicons_iconv/"${file}")
done
