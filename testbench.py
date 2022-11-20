# import language_evaluation
# from pprint import PrettyPrinter
# pprint = PrettyPrinter().pprint

# predicts = ['i am a boy', 'she is a girl']
# answers = ['am i a boy ?', 'is she a girl ?']

# evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])
# results = evaluator.run_evaluation(predicts, answers)
# print(results)


import torch
import torch.nn as nn