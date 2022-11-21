#!/bin/bash


for ((i=12; i<=18; i=i+4)); do
    python eval3_2_all_copy.py $1 $2 --resume_name="epoch_${i}_best.pth"
    python p2_evaluate.py --pred_file=$2
done
# python eval3_2_all_copy.py $1 $2 
# python p2_evaluate.py --pred_file=$2
     
# bash eval3_2.sh hw3_data/p2_data/images/val hw3/output_p2/pred.json
#bash eval3_2.sh hw3_data/p2_data/images/val hw3/output_p2/pred.json  >> ./hw3/output_p2/log.txt