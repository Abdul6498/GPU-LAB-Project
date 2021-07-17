# GPU-LAB-Project

Output log on macOS:

```
(files not shown)
Select Left image index: 11
Select Right image index: 9
SAD Processing started...........
NCC Processing started...........
SAD Processing Completed
NCC Processing Completed
Median
Median
Ignoring AMD Radeon Compute Engine because the compute kernel crashes macOS
Running on Intel(R) Iris(TM) Plus Graphics 655
[CPU] Histogram Execution Time: 2ms
[CPU] SAD Execution Time: 1605ms
[CPU] NCC Execution Time: 2604ms
[CPU] Median Execution Time: 293ms
 --- 
[GPU] [SAD] Upload Time: 0.000033s
[GPU] [SAD] Equalization Execution Time: 0.000000s
[GPU] [SAD] Disparity Execution Time: 0.015434s
[GPU] [SAD] Denoising Time: 0.006361s
[GPU] [SAD] Download Time: 0.000012s
[GPU] [SAD] Total Time: 0.021840s
[GPU] [NCC] Upload Time: 0.000034s
[GPU] [NCC] Equalization Execution Time: 0.000000s
[GPU] [NCC] Disparity Execution Time: 0.015130s
[GPU] [NCC] Denoising Time: 0.006662s
[GPU] [NCC] Download Time: 0.000012s
[GPU] [NCC] Total Time: 0.021838s
 --- 
[SAD] GPU speedup: 103.991188
[SAD] GPU speedup (+overhead): 80.128205
[NCC] GPU speedup: 172.108394
[NCC] GPU speedup (+overhead): 126.018866

Process finished with exit code 0
```
