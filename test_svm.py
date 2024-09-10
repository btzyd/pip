
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib
import argparse
from pathlib import Path

def print_markdown(prefix, result):
    print(prefix)
    print("| | precision | recall | f1-score | support |")
    print("| ---: | :---: | :---: | :---: | :---: |")
    print("| clean | {:.2%} | {:.2%} | {:.2%} | {:.0f} |".format(result["clean"]["precision"], result["clean"]["recall"], result["clean"]["f1-score"], result["clean"]["support"]))
    print("| adversarial | {:.2%} | {:.2%} | {:.2%} | {:.0f} |".format(result["adversarial"]["precision"], result["adversarial"]["recall"], result["adversarial"]["f1-score"], result["adversarial"]["support"]))
    print("| accuracy | - | - | {:.2%} | {:.0f} |".format(result["accuracy"], result["macro avg"]["support"]))
    print("| macro avg | {:.2%} | {:.2%} | {:.2%} | {:.0f} |".format(result["macro avg"]["precision"], result["macro avg"]["recall"], result["macro avg"]["f1-score"], result["macro avg"]["support"]))
    print("| weighted avg | {:.2%} | {:.2%} | {:.2%} | {:.0f} |".format(result["weighted avg"]["precision"], result["weighted avg"]["recall"], result["weighted avg"]["f1-score"], result["weighted avg"]["support"]))
    print("---")

def main(args):
    attention_map_dir = os.path.join(args.work_dir, "attention_map")

    attention_map_list = os.listdir(attention_map_dir)

    attention_map = []
    y_true = np.zeros(len(attention_map_list))
    for i, attention_file in enumerate(attention_map_list):
        attention_map.append(np.load(os.path.join(attention_map_dir, attention_file)))
        if "adv" in attention_file:
            y_true[i] = 1
    attention_map = np.stack(attention_map, axis=1)
    question_num = np.size(attention_map, 0)
    sample_num = np.size(attention_map, 1)
    attention_map = attention_map.reshape(question_num, sample_num, -1)

    for i in range(args.svm_total_num):
        svm_classifier = joblib.load(os.path.join(args.svm_dir, f"svm_question_{i}.pkl"))
        if i==0:
            y_pred = svm_classifier.predict(attention_map[i])
        else:
            y_pred += svm_classifier.predict(attention_map[i])

    y_pred = np.where(y_pred >= args.svm_alarm_num, 1, 0)
    
    result_1000_1000 = classification_report(y_true, y_pred, digits=4, output_dict=True, target_names=["clean", "adversarial"])
    print_markdown("work_dir: {}, svm_dir: {}, 1000:1000".format(args.work_dir, args.svm_dir), result_1000_1000)

    np.random.seed(0)
    mask_zeros = (y_true == 0)
    ones_indices = np.where(y_true == 1)[0]
    random_ones_indices = np.random.choice(ones_indices, 100, replace=False)
    mask_ones = np.zeros_like(y_true, dtype=bool)
    mask_ones[random_ones_indices] = True
    final_mask = mask_zeros | mask_ones
    
    result_1000_100 = classification_report(y_true[final_mask], y_pred[final_mask], digits=4, output_dict=True, target_names=["clean", "adversarial"])
    print_markdown("work_dir: {}, svm_dir: {}, 1000:100, svm_alarm_num: {}, svm_total_num: {}".format(args.work_dir, args.svm_dir, args.svm_alarm_num, args.svm_total_num), result_1000_100)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--svm_dir', required=True)
    parser.add_argument('--svm_alarm_num', default=1, type=int)
    parser.add_argument('--svm_total_num', default=1, type=int)
    args = parser.parse_args()
    main(args)
