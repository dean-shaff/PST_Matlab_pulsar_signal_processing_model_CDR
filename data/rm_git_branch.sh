#! /bin/bash 

for filename in *.dean-devel.*; do 
	mv "${filename}" "${filename//.dean-devel/}" 
done 
