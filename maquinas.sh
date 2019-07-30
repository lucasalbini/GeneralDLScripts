#!/bin/sh
rm `hostname`.pdf
mpstat -P ALL > output
tail -n +3 output > cpus
top -b -n1 > top.txt
getent passwd | grep /home/users | cut -d":" -f1 > usuarios.txt
head -7 top.txt > cabtotal
sed '3d' cabtotal > cab
cat top.txt | grep -f usuarios.txt > corpo
cat cpus cab corpo > `hostname`.txt
enscript -f "Times-Roman14" --encoding=88591 -p file.html `hostname`.txt
ps2pdf file.ps `hostname`.pdf

#rm file.ps
rm output
rm cpus
rm top.txt
rm usuarios.txt
rm cabtotal
rm cab
rm corpo
#rm `hostname`.txt
