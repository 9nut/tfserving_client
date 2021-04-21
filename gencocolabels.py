#! /usr/bin/env python3

# Using the COCO 2017 validation dataset annotations, generate
# a list of class names in the order of class id.

import argparse
import json

def cococlasses(annotpath):
    with open(annotpath,'r') as cocoinfo:
        infojson = json.loads(cocoinfo.read())
        categories = infojson['categories']
        classnames = ['N/A' for i in range(91)]
        for cat in categories:
            classnames[cat['id']] = cat['name']

    # print(classnames)
    return classnames

def main():
    parser = argparse.ArgumentParser(prog='gencocolabels', description='Discover all COCO labels.')
    parser.add_argument('--path', dest='path', type=str, default='/data/Datasets/COCO/annotations/instances_val2017.json', help='location of the label file')
    args = parser.parse_args()

    args = parser.parse_args()
    labels = cococlasses(args.path)

    with open("COCOlabels.txt", 'w') as output:
        for x in labels:
            output.write("{}\n".format(x))

if __name__ == '__main__':
    main()