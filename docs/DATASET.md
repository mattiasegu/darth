## Symlink
We recommend to symlink all the additional directories containing large files to the main project directory `$PROJECT_DIR/`. For example, the dataset root to `./data`, the checkpoints root
to `./checkpoints`, and the work directory to `./work_dirs`.

This will avoid storing large files in your project directory, a requirement of several high-performance computing systems.

For each directory `$LARGE_DIR`, symlink it to the `darth` base directory using `ln -s $LARGE_DIR/ $PROJECT_DIR/`.

## Dataset
Download the datasets from their official website to your `$DATADIR`. 

Symlink your data directory to the `darth` base directory using:
```shell
ln -s $DATADIR/ $PROJECT_DIR/
```

Check out [SHIFT.md](docs/SHIFT.md) for instructions on how to download and process the [SHIFT](https://www.vis.xyz/shift/) dataset.

### Convert Annotations
Following `mmtracking`, we use [CocoVID](https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/datasets/parsers/coco_video_parser.py) to maintain all datasets in this codebase.
In this case, you need to convert the official annotations to this style. We provide scripts and the usages are as following:

```shell
# MOT17
# The processing of other MOT Challenge dataset is the same as MOT17
python ./tools/convert_datasets/mot/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
python ./tools/convert_datasets/mot/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3

# DanceTrack
python ./tools/convert_datasets/dancetrack/dancetrack2coco.py -i ./data/dancetrack -o ./data/dancetrack/annotations

# BDD100k
## add location attributes to detection labels (if needed)
python ./tools/convert_datasets/bdd100k/location_to_det.py -n 8 -i ./data/bdd100k/info/100k/ -d ./data/bdd100k/labels/det_20/ -o ./data/bdd100k/labels/det_20_with_locations/
## change naming (only if previous step was executed)
mv ./data/bdd100k/labels/det_20/ ./data/bdd100k/labels/det_no_locations/
mv ./data/bdd100k/labels/det_20_with_locations/ ./data/bdd100k/labels/det_20/
## add domain attributes to tracking labels
python ./tools/convert_datasets/bdd100k/attributes_to_track.py -n 8 -d ./data/bdd100k/labels/det_20/ -t ./data/bdd100k/labels/box_track_20/ -o ./data/bdd100k/labels/box_track_20_with_domains/
## change naming
mv ./data/bdd100k/labels/box_track_20/ ./data/bdd100k/labels/box_track_20_no_domains/
mv ./data/bdd100k/labels/box_track_20_with_domains/ ./data/bdd100k/labels/box_track_20/
## detection
mkdir -p data/bdd100k/annotations/det_20
python -m bdd100k.label.to_coco -m det -i ./data/bdd100k/labels/det_20/det_${SET_NAME}.json -o ./data/bdd100k/annotations/det_20/box_det_${SET_NAME}_cocofmt.json
## tracking
mkdir -p data/bdd100k/annotations/box_track_20
python -m bdd100k.label.to_coco -m box_track -i ./data/bdd100k/labels/box_track_20/${SET_NAME} -o ./data/bdd100k/annotations/box_track_20/box_track_${SET_NAME}_cocofmt.json

# SHIFT
## detection/tracking
mkdir -p data/shift/discrete/videos/${SET_NAME}/front/
python -m scalabel.label.to_coco -m box_track -i ./data/shift/discrete/videos/${SET_NAME}/front/det_2d.json -o ./data/shift/discrete/videos/${SET_NAME}/front/det_2d_cocofmt.json
```

The `${SET_NAME}` here can be one of ['train', 'val'].

### Folder Structure
We here report our folder structure. If your folder structure is different, you may need to change the corresponding paths in config files.

```
├── checkpoints
├── configs
├── data
│   ├── bdd
│   │   ├── images 
│   │   │   ├── 100k 
|   |   |   |   |── train
|   |   |   |   |── val
│   │   │   ├── track 
|   |   |   |   |── train
|   |   |   |   |── val
│   │   ├── annotations 
│   │   │   ├── box_track_20
│   │   │   ├── det_20
│   │   ├── labels 
│   │   │   ├── box_track_20
│   │   │   ├── box_track_20_no_domains
│   │   │   ├── det_20
│   ├── crowdhuman
│   │   ├── annotations 
│   │   ├── val 
│   │   ├── train
│   ├── dancetrack
│   │   ├── annotations 
│   │   ├── test 
│   │   ├── train
│   │   ├── val
│   ├── MOT17
│   │   ├── annotations 
│   │   ├── reid 
│   │   ├── test 
│   │   ├── train
│   ├── shift
│   │   ├── discrete 
│   │   │   ├── videos
│   │   │   │   ├── train
│   │   │   │   │   ├── front
│   │   │   │   ├── val
│   │   │   │   │   ├── front
│   .
│   .
│   .
├── darth
├── demo
├── docs
├── resources
├── scripts
├── test
├── tools
├── work_dirs
```


