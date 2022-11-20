#!/bin/bash


# for ((i=5; i<=18; i++)); do
#     python eval3_2_zarzouram.py $1 $2 --resume_name="epoch_${i}_best.pth"
#     python p2_evaluate.py --pred_file=$2
# done
python eval3_3.py hw3_data/p2_data/images/val hw3/output_p2/pred.json
     
# bash eval3_3.sh