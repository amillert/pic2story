$(which python) ./src/main.py -p ./data/pics -l ./data/yolo_objects/coco.names -n ./data/yolo_objects/yolo.cfg -t ./data/yolo_objects/yolov3.weights -c 0.8 -b ./data/gutten/morgen/ --save_corpus ./cache/corpus_big_500.pkl --read_corpus ./cache/corpus_big_500.pkl --batch_size 64 --epochs 20 --eta 2E-3 -g 3 --hidden 30 --layers 12 --drop_prob 0.3 --window 8
