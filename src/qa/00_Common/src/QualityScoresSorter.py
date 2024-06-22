# Shree KRISHNAya Namaha
# Sorts keys in QualityScores.json files
# Author: Nagabhushan S N
# Last Modified: 10/01/2024
import json
import time
import datetime
import traceback
import pandas

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def sort_dict(unsorted_dict: dict):
    sorted_dict = {}
    for k, v in sorted(unsorted_dict.items()):
        if isinstance(v, dict):
            sorted_dict[k] = sort_dict(v)
        else:
            sorted_dict[k] = v
    return sorted_dict


def sort_qa_scores(training_dirpath: Path, train_nums: list):
    for train_num in train_nums:
        qa_filepath = training_dirpath / f'train{train_num:04}/QualityScores.json'
        with open(qa_filepath, 'r') as qa_file:
            qa_dict = json.load(qa_file)
        sorted_qa_dict = sort_dict(qa_dict)
        with open(qa_filepath, 'w') as qa_file:
            json.dump(sorted_qa_dict, qa_file, indent=4)
    return


def demo1():
    training_dirpath = Path('../../../view_synthesis/research/011_SparseDepthPrior/runs/training')
    train_nums = [3]
    sort_qa_scores(training_dirpath, train_nums)
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
