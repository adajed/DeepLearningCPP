#ifndef GRAPHDL_CORE_LAYERS_CUDA_MACROS_H_
#define GRAPHDL_CORE_LAYERS_CUDA_MACROS_H_

#define POS_2D(x1, x2, shape) \
    ((x1) * (shape)[2] + (x2))

#define POS_3D(x1, x2, x3, shape) \
    (((x1) * (shape)[1] + (x2)) * (shape)[2] + (x3))

#define POS_4D(x1, x2, x3, x4, shape) \
    ((((x1) * (shape)[1] + (x2)) * (shape)[2] + (x3)) * (shape)[3] + (x4))

#endif
