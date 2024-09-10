
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib
import argparse
from pathlib import Path

def main(args):
    attention_map_dir = os.path.join(args.work_dir, "attention_map")

    attention_map_list = os.listdir(attention_map_dir)

    attention_map = []
    svm_y = np.zeros(len(attention_map_list))
    for i, attention_file in enumerate(attention_map_list):
        attention_map.append(np.load(os.path.join(attention_map_dir, attention_file)))
        if "adv" in attention_file:
            svm_y[i] = 1
    attention_map = np.stack(attention_map, axis=1)
    question_num = np.size(attention_map, 0)
    sample_num = np.size(attention_map, 1)
    attention_map = attention_map.reshape(question_num, sample_num, -1)

    if args.question_index==-1:
        for i in range(question_num):
            print(f"training the {i}/{question_num} svm...")
            svm_classifier = SVC(kernel='linear', C=10)
            svm_classifier.fit(attention_map[i], svm_y)
            joblib.dump(svm_classifier, os.path.join(args.svm_dir, f"svm_question_{i}.pkl"))
    else:
        print(f"training the {args.question_index} svm...")
        svm_classifier = SVC(kernel='linear', C=10)
        svm_classifier.fit(attention_map[args.question_index], svm_y)
        joblib.dump(svm_classifier, os.path.join(args.svm_dir, f"svm_question_{args.question_index}.pkl"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--svm_dir', default="svm")
    parser.add_argument('--question_index', default=-1, type=int)
    args = parser.parse_args()

    args.svm_dir = os.path.join(args.work_dir, args.svm_dir)
    Path(args.svm_dir).mkdir(parents=True, exist_ok=True)

    main(args)
