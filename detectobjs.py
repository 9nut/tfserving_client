#! /usr/bin/env python3

import argparse

import detclient.cgrpc.detector as gdetcli
import detclient.crest.detector as rdetcli

class CenterNetGRPC(gdetcli.Detector):
    def __init__(self, hostport):
        super().__init__(hostport, 'centernet_hourglass_512x512_kpts', './COCOlabels.txt')

class CenterNetREST(rdetcli.Detector):
    def __init__(self, hostport):
        super().__init__(hostport, 'centernet_hourglass_512x512_kpts', './COCOlabels.txt')

def main():
    parser = argparse.ArgumentParser(prog='detectobj', description='Perform object detections.')
    parser.add_argument('images', nargs='+', type=str, help='image1.jpg [image2.jpg ...]')
    parser.add_argument('--label', dest='label', type=str, help='label name to select (COCO2017)')
    parser.add_argument('--rest', default=False, action='store_true', help='Use REST Client')
    args = parser.parse_args()

    if args.rest:
        dc = CenterNetREST('localhost:8501')
    else:
        dc = CenterNetGRPC('localhost:8500')

    for image in args.images:
        print("{}:{}".format(image, dc.predict(image, label=args.label)))


if __name__ == '__main__':
    main()
