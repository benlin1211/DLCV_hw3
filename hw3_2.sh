#!/bin/bash

# TODO - run your inference Python3 code


# python eval3_2_nn_transformer.py $1 $2
# python eval3_2_zarzouram.py $1 $2
# python eval3_2_all_copy.py $1 $2
python eval3_2_all_copy_bean.py $1 $2
# python p2_evaluate.py --pred_file=hw3/output_p2/pred.json

# bash hw3_2.sh hw3_data/p2_data/images/val hw3/output_p2/pred.json >> ./hw3/output_p2/log.txt