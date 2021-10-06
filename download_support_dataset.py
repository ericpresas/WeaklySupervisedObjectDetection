# https://github.com/cocodataset/cocoapi
from data import COCOSupportSet

if __name__ == "__main__":
    data_path = "data/coco"
    data_type = 'train2017'
    categories = ['person', 'dog', 'skateboard']
    coco_dataset = COCOSupportSet(
        data_path=data_path,
        data_id='train2017',
        categories=categories,
        max_samples_category=20
    )

    category_names = coco_dataset.list_category_names()

    coco_dataset.categories = category_names[0:20]

    coco_dataset.save_images()
    print('Done.')