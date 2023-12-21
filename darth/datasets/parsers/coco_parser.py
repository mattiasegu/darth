"""Parse for COCO-like image datasets."""
from collections import defaultdict
import itertools

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

from mmdet.datasets.api_wrappers import COCO as _COCO


class COCO(_COCO):
    """Inherit mmdet COCO class to parse the annotations of bbox-related
    image tasks.

    It implements image filtering based on allowed attributes and annotation
    filtering based on classes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attributes_to_imgs = self.attributesToImgs
        self.cat_name_to_id = self.catNameToId

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs, catNameToId = {}, {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        attributesToImgs = defaultdict(lambda: defaultdict(list))
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
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

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        self.attributesToImgs = attributesToImgs
        self.catNameToId = catNameToId 

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[], attributes=None):
        return self.getImgIds(img_ids, cat_ids, attributes)

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]

        return ids

    def getImgIds(self, imgIds=[], catIds=[], attributes=None):
        '''
        Get img ids that satisfy given filter conditions.
        Args:
            imgIds (list[int]): The given image ids. Defaults to [].
            catIds (list[int]): Get images with all given cats. Defaults to [].
            attributes (Optional[Dict[List[any]]]): a dictionary of allowed
                attributes according to which videos are filtered. If None,
                do not filter. 
        Return:
            ids (int array), an integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0 and not attributes:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
            if attributes is not None:
                for i, (key, value) in enumerate(attributes.items()):
                    if i == 0 and len(ids) == 0 and len(catIds) == 0:
                        ids = set(self.attributes_to_imgs[key][value])
                    else:
                        ids &= set(self.attributes_to_imgs[key][value])

        return list(ids)

