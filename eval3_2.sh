#!/bin/bash


# for ((i=5; i<=18; i++)); do
#     python eval3_2_zarzouram.py $1 $2 --resume_name="epoch_${i}_best.pth"
#     python p2_evaluate.py --pred_file=$2
# done
python eval3_2_all_copy.py $1 $2 
python p2_evaluate.py --pred_file=$2
     
# bash eval3_2.sh hw3_data/p2_data/images/val hw3/output_p2/pred.json  >> ./hw3/output_p2/train2-1_tutorial_A.txt
# bash eval3_2.sh hw3_data/p2_data/images/val hw3/output_p2/pred.json