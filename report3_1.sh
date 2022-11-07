#!/bin/bash

echo "p1"
python eval3_1.py hw3_data/p1_data/val ./hw3_data/p1_data/id2label.json "./output_p1/pred1.csv" --template_prompt="This is a photo of {object} "
python acc3_1.py --csv_path="./output_p1/pred1.csv"
echo "p2"
python eval3_1.py hw3_data/p1_data/val ./hw3_data/p1_data/id2label.json "./output_p1/pred2.csv" --template_prompt="This is a {object} image. "
python acc3_1.py --csv_path="./output_p1/pred2.csv"
echo "p3"
python eval3_1.py hw3_data/p1_data/val ./hw3_data/p1_data/id2label.json "./output_p1/pred3.csv" --template_prompt="No {object}, no score. "
python acc3_1.py --csv_path="./output_p1/pred3.csv"