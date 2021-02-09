import os
import argparse
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_dir', type=str, default='./data/Market-1501-v15.09.15/pytorch')
    return parser.parse_args()


def main():
    args = get_args()
    class_map = {}
    cnt = 0
    if args.mode == 'train':
        dir = '%s/train' % args.dataset_dir
        for dir in os.listdir(dir):
            if len(dir) == 4:
                if dir not in class_map:
                    class_map[dir] = cnt
                    cnt += 1
        with open('cache/train_class_map.pkl', 'wb') as fb:
            pickle.dump(class_map, fb)
    if args.mode == 'test':
        dir = '%s/query' % args.dataset_dir
        for dir in os.listdir(dir):
            if len(dir) == 4:
                if dir not in class_map:
                    class_map[dir] = cnt
                    cnt += 1
        with open('cache/test_class_map.pkl', 'wb') as fb:
            pickle.dump(class_map, fb)


if __name__ == '__main__':
    main()