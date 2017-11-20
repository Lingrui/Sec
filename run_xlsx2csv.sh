for file in /home/lcai/s2/sec_10k/raw_data/*.xlsx 
do 
    out=$(echo ${file} | sed 's/xlsx$/csv/' | sed 's/raw_data/preprocess/') 
    python  ./xlsx2csv.py ${file} ${out}
done
