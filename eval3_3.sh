#!/bin/bash


# for ((i=5; i<=18; i++)); do
#     python eval3_2_zarzouram.py $1 $2 --resume_name="epoch_${i}_best.pth"
#     python p2_evaluate.py --pred_file=$2
# done
python eval3_3.py 
python eval3_3_all.py
     
# bash eval3_3.sh