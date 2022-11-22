#!/bin/bash


# for ((i=5; i<=18; i++)); do
#     python eval3_2_zarzouram.py $1 $2 --resume_name="epoch_${i}_best.pth"
#     python p2_evaluate.py --pred_file=$2
# done
python eval3_3.py 
python eval3_3_find_pairs.py --des_root="hw3/output_p3-2_data/"
# 手動把那兩張移過去?
python eval3_3_report.py --src_path="hw3/output_p3-2_data/" --des_root="hw3/output_p3-2/"
# bash eval3_3.sh