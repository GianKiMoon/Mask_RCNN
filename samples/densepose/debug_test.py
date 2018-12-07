import densepose as dp

dp.main(["train", "--dataset", "..\..\datasets\coco", "--weights", "last"])
#dp.main(["inference", "--dataset", "..\..\datasets\coco", "--weights", "coco"])