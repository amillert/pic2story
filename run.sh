./src/main.py -p ./data/pics -l ./utils/coco.names -n ./utils/yolo.cfg -t ./utils/yolov3.weights -c 0.8 -b ./data/gutten/morgen/ --ngrams 2 --save_corpus ./cache/corpus_big_500.pkl --read_corpus ./cache/corpus_big_500.pkl --batch_size 64 --epochs 20 --eta 2E-3 -g 3 --hidden 30 --layers 12 --drop_prob 0.3
