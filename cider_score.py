import language_evaluation
import torch
from pprint import PrettyPrinter


def getCIDErScore(predicts, answers):
    evaluator = language_evaluation.CocoEvaluator()
    results = evaluator.run_evaluation(predicts, answers)
    return 0 #results

if __name__ =="__main__":
    pprint = PrettyPrinter().pprint
    predicts = ['i am a boy', 'she is a girl']
    answers = ['am i a boy ?', 'is she a girl ?']
    evaluator = language_evaluation.CocoEvaluator()
    results = evaluator.run_evaluation(predicts, answers)
    print(results)
    #results = getCIDErScore(predicts, answers)
    
