import cv2 as cv
import numpy as np
import sys

if __name__ == '__main__':
    model = 'resources/model.yml.gz'
    im = cv.imread('data/coco/detection/train/airplane/6.jpg')

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(30)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    for b in boxes[0]:
        x, y, w, h = b
        cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    cv.imshow("edges", edges)
    cv.imshow("edgeboxes", im)
    cv.waitKey(0)
    cv.destroyAllWindows()
