# https://github.com/cocodataset/cocoapi
from data import COCODatasetDetection

if __name__ == "__main__":
    data_path = "data/coco"
    data_type = 'train2017'
    categories = ['person', 'dog', 'skateboard']
    coco_dataset = COCODatasetDetection(
        data_path=data_path,
        data_train_id='train2017',
        data_val_id='val2017',
        categories=categories,
        max_samples_category=80
    )

    category_names = coco_dataset.list_category_names()

    coco_dataset.categories = category_names[0:20]

    coco_dataset.build_dataset_splits()



    coco_dataset.save_images('train')
    coco_dataset.save_images('val')
    coco_dataset.save_images('test')
    print('Done.')