"""Parser for COCO-like video datasets."""
from collections import defaultdict

import numpy as np
from pycocotools.coco import _isArrayLike

from . import COCO

class CocoVID(COCO):
    """Inherit official COCO class in order to parse the annotations of bbox-
    related video tasks.

    Args:
        annotation_file (str): location of annotation file. Defaults to None.
        load_img_as_vid (bool): If True, convert image data to video data,
            which means each image is converted to a video. Defaults to False.
    """

    def __init__(self, annotation_file=None, load_img_as_vid=False):
        assert annotation_file, 'Annotation file must be provided.'
        self.load_img_as_vid = load_img_as_vid
        super(CocoVID, self).__init__(annotation_file=annotation_file)
        self.attributes_to_vids = self.attributesToVids
        self.cat_to_vids = self.catToVids

    def convert_img_to_vid(self, dataset):
        """Convert image data to video data."""
        if 'images' in self.dataset:
            videos = []
            for i, img in enumerate(self.dataset['images']):
                videos.append(
                    dict(id=img['id'],
                    name=img['file_name'],
                    attributes=img.get('attributes', None))
                )
                img['video_id'] = img['id']
                img['frame_id'] = 0
            dataset['videos'] = videos

        if 'annotations' in self.dataset:
            for i, ann in enumerate(self.dataset['annotations']):
                ann['video_id'] = ann['image_id']
                ann['instance_id'] = ann['id']
        return dataset

    def createIndex(self):
        """Create index."""
        print('creating index...')
        anns, cats, imgs, vids, catNameToId = {}, {}, {}, {}, {}
        (
            imgToAnns, catToImgs, vidToImgs, vidToInstances, instancesToImgs
        ) = (
            defaultdict(list), defaultdict(list), defaultdict(list),
            defaultdict(list), defaultdict(list)
        )

        attributesToVids = defaultdict(lambda: defaultdict(list))
        attributesToImgs = defaultdict(lambda: defaultdict(list))
        catToVids = defaultdict(set)

        if 'videos' not in self.dataset and self.load_img_as_vid:
            self.dataset = self.convert_img_to_vid(self.dataset)

        if 'videos' in self.dataset:
            for video in self.dataset['videos']:
                vids[video['id']] = video
                if 'attributes' in video:
                    for key, value in video['attributes'].items():
                        attributesToVids[key][value].append(video['id']) 

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann
                if 'instance_id' in ann:
                    instancesToImgs[ann['instance_id']].append(ann['image_id'])
                    if 'video_id' in ann and \
                        ann['instance_id'] not in \
                            vidToInstances[ann['video_id']]:
                        vidToInstances[ann['video_id']].append(
                            ann['instance_id'])

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                vidToImgs[img['video_id']].append(img)
                imgs[img['id']] = img
                if 'attributes' in img:
                    for key, value in img['attributes'].items():
                        attributesToImgs[key][value].append(img['id'])

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
                catNameToId[cat['name']] = cat['id']

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])
                if 'images' in self.dataset:
                    catToVids[ann['category_id']].add(imgs[ann['image_id']
                    ]['video_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        self.videos = vids
        self.vidToImgs = vidToImgs
        self.vidToInstances = vidToInstances
        self.instancesToImgs = instancesToImgs

        self.attributesToImgs = attributesToImgs 
        self.attributesToVids = attributesToVids 
        self.catToVids = catToVids
        self.catNameToId = catNameToId 

    def get_vid_ids(self, vid_ids=[], cat_ids=[], attributes=None):
        """Get video ids that satisfy given filter conditions.

        Default return all video ids.

        Args:
            vid_ids (list[int]): The given video ids. Defaults to [].
            cat_ids (list[int]): Get videos with all given cats. Defaults to [].
            attributes (Optional[Dict[List[any]]]): a dictionary of allowed
                attributes according to which videos are filtered. If None,
                do not filter. 

        Returns:
            list[int]: Video ids.
        """
        vid_ids = vid_ids if _isArrayLike(vid_ids) else [vid_ids]

        if len(vid_ids) == 0 and len(cat_ids) == 0 and not attributes:
            ids = self.videos.keys()
        else:
            ids = set(vid_ids)
            for i, cat_id in enumerate(cat_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(self.cat_to_vids[cat_id])
                else:
                    ids &= set(self.cat_to_vids[cat_id])
            if attributes is not None:
                for i, (key, value) in enumerate(attributes.items()):
                    if i == 0 and len(ids) == 0 and len(cat_ids) == 0:
                        ids = set(self.attributes_to_vids[key][value])
                    else:
                        ids &= set(self.attributes_to_vids[key][value])

        return list(ids)

    def get_img_ids_from_vid(self, vidId):
        """Get image ids from given video id, supporting videos with frame_ids
        not starting from 0.

        Args:
            vidId (int): The given video id.

        Returns:
            list[int]: Image ids of given video id.
        """
        img_infos = self.vidToImgs[vidId]
        ids = list(np.zeros([len(img_infos)], dtype=np.int64))
        img_infos = sorted(img_infos, key=lambda i: i['frame_id'])
        first_id = img_infos[0]['frame_id'] 
        for i, img_info in enumerate(img_infos):
            ids[i] = img_info['id']
            # frame_ids must start from 0
            img_info['frame_id'] -= first_id
        return ids

    def get_ins_ids_from_vid(self, vidId):
        """Get instance ids from given video id.

        Args:
            vidId (int): The given video id.

        Returns:
            list[int]: Instance ids of given video id.
        """
        return self.vidToInstances[vidId]

    def get_img_ids_from_ins_id(self, insId):
        """Get image ids from given instance id.

        Args:
            insId (int): The given instance id.

        Returns:
            list[int]: Image ids of given instance id.
        """
        return self.instancesToImgs[insId]

    def load_vids(self, ids=[]):
        """Get video information of given video ids.

        Default return all videos information.

        Args:
            ids (list[int]): The given video ids. Defaults to [].

        Returns:
            list[dict]: List of video information.
        """
        if _isArrayLike(ids):
            return [self.videos[id] for id in ids]
        elif type(ids) == int:
            return [self.videos[ids]]

    
