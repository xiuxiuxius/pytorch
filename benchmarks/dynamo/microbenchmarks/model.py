# resnet50 layer shape
resnet50_layers = (
    # IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding
    (224, 224, 3, 7, 7, 64, (2, 2), (0, 0)),
    # conv2_x
    (56, 56, 64, 1, 1, 64, (1, 1), (0, 0)),
    (56, 56, 64, 3, 3, 64, (1, 1), (0, 0)),
    (56, 56, 64, 1, 1, 256, (1, 1), (0, 0)),
    # conv3_x
    (56, 56, 256, 1, 1, 128, (2, 2), (0, 0)),
    (28, 28, 128, 3, 3, 128, (1, 1), (0, 0)),
    (28, 28, 128, 1, 1, 512, (1, 1), (0, 0)),
    # conv4_x
    (28, 28, 512, 1, 1, 256, (2, 2), (0, 0)),
    (14, 14, 256, 3, 3, 256, (1, 1), (0, 0)),
    (14, 14, 256, 1, 1, 1024, (1, 1), (0, 0)),
    # conv5_x
    (14, 14, 1024, 1, 1, 512, (2, 2), (0, 0)),
    (7, 7, 512, 3, 3, 512, (1, 1), (0, 0)),
    (7, 7, 512, 1, 1, 2048, (1, 1), (0, 0)),
)

alexnet_layers = (
    # IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding
    (224, 224, 3, 11, 11, 64, (4, 4), (2, 2)),
)
