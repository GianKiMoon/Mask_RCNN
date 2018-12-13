import densepose as dp

dp.main(["train", "--dataset", "..\..\datasets\coco", "--weights", "coco"])
#dp.main(["inference", "--dataset", "..\..\datasets\coco", "--weights", "coco"])