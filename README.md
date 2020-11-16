## General idea behind the Pic2Story
`Pic2Story` is the idea of an NLP tool which, given a set of images, provides the user with generated text - a type of story inspired by the inputted pictures.

The emphasis is put on the latter part of the project; therefore, the image object recognition aspect is not to be considered the core of the application (`YOLO` has been utilized).

### The main points of interest are:
- creating an English story-based corpus,
- developing a tool for automatic text generation,
- utilizing the text-generation tool based on the corpus, alongside the annotated images with the detected object.

### To be considered in further stages:
- custom object detection,
- deployment of the app as a web application to allow accessability for the online users.

## Install Pic2Story
Download the source code from github: 
``` bash
git clone https://github.com/amillert/pic2story.git
```

## Running
In order to properly run all the commands one must ensure that the current directory is `pic2story/`:
``` bash
cd pic2story/
```

### Install YOLO
The script installs YOLO and downloads required weights for the model
``` bash
./data/yolo_objects/install-YOLO.sh
```

### Download Textual Books
The dataset comes from the [Project Gutenberg's](https://www.gutenberg.org/) website. In order to download the data and unzip recursively located `*.txt` files, one should run the follwoing command:
``` bash
./data/gutten/getgutenbergdata.sh
```
The following script will first download the data, then run unzipping python script.

### Create and Activate Conda Virtual Environment

``` bash
conda env create -f pic2story.yml; conda activate pic2story
```

### Script's Help
In order to find out about the arguements required to successfully run the scrip, one should run:
``` bash
./scr/main.py -h
```

### Run Pic2Story
In order to run the whole application with sample parameters, one should run:
``` bash
./run.sh
```
