from mrcnn import model as mrcnn_model
from samples.densepose import densepose as dp
import synthpod.model as dp_model

mask_rcnn_weights = ''
densepose_branch_weights = ''

if __name__ == "__main__":
    dp.main(["inference", "--dataset", "..\..\datasets\coco", "--weights", "coco"])

    model_densepose = dp_model.ResNet(nClasses=15,
                            input_height=256,
                            input_width=256)