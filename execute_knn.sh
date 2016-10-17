#!/bin/bash 

IFS=$'\n'
LIST='/media/ezetl/0C74D0DD74D0CB1A/Datasets/Faces/imdbwiki_fv/imdbwiki_fvpathssss.txt'
REPORT='report.org'
TMP='tmp.txt'

# save first tree
first_file=$(head -1 $LIST | awk '{print $1}')
./search knn $LIST  $first_file 'save'


# compute more images
i=0
for file in $(head -50 $LIST | awk '{print $1}')
do 
    echo "*** Example $i" >> $REPORT
    echo -e "#+begin_src python :results output\n***" >> $REPORT
    ./search knn $LIST  $file 'load' &>> $TMP
    cat $TMP | sed 's/\/media\/ezetl\/0C74D0DD74D0CB1A\/Datasets\/Faces\/imdbwiki_fv\///g' >> $REPORT
    echo -e "***\n#+end_src" >> $REPORT
    echo -e "#+CAPTION: Example $i\n#+NAME: ex$i" >> $REPORT
    rgrep=$(grep "Result saved at" $TMP | awk '{print $4}')
    echo -e "[[$rgrep]]" >> $REPORT
    echo -e "\n\n\n" >> $REPORT 
    i=$((i + 1))
    rm $TMP
done
