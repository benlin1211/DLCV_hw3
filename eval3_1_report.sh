#!/bin/bash

echo "P2 baseline"
python eval3_1.py hw3_data/p1_data/val ./hw3_data/p1_data/id2label.json "./output_p1/pred_base.csv" --template_prompt="{object}"
python acc3_1.py --csv_path="./output_p1/pred_base.csv"
echo "P2 prompt 1"
python eval3_1.py hw3_data/p1_data/val ./hw3_data/p1_data/id2label.json "./output_p1/pred1.csv" --template_prompt="This is a photo of {object} "
python acc3_1.py --csv_path="./output_p1/pred1.csv"
echo "P2 prompt 2"
python eval3_1.py hw3_data/p1_data/val ./hw3_data/p1_data/id2label.json "./output_p1/pred2.csv" --template_prompt="This is a {object} image. "
python acc3_1.py --csv_path="./output_p1/pred2.csv"
echo "P2 prompt 3"
python eval3_1.py hw3_data/p1_data/val ./hw3_data/p1_data/id2label.json "./output_p1/pred3.csv" --template_prompt="No {object}, no score. "
python acc3_1.py --csv_path="./output_p1/pred3.csv"
echo "P3"
python eval3_1-3.py hw3_data/p1_data/val ./hw3_data/p1_data/id2label.json "./output_p1/"
# bash report3_1.sh