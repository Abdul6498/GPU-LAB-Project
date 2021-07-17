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
[CPU] SAD Execution Time: 1261ms
[CPU] NCC Execution Time: 2251ms
[CPU] Median Execution Time: 280ms
 --- 
[GPU] [SAD] Upload Time: 0.000034s
[GPU] [SAD] Equalization Execution Time: 0.000000s
[GPU] [SAD] Disparity Execution Time: 0.015507s
[GPU] [SAD] Denoising Time: 0.006337s
[GPU] [SAD] Download Time: 0.000013s
[GPU] [SAD] Total Time: 0.021891s
[GPU] [NCC] Upload Time: 0.000034s
[GPU] [NCC] Equalization Execution Time: 0.000000s
[GPU] [NCC] Disparity Execution Time: 0.015349s
[GPU] [NCC] Denoising Time: 0.006627s
[GPU] [NCC] Download Time: 0.000012s
[GPU] [NCC] Total Time: 0.022022s
 --- 
[SAD] GPU speedup: 81.318114
[SAD] GPU speedup (+overhead): 63.953223
[NCC] GPU speedup: 146.654505
[NCC] GPU speedup (+overhead): 108.618654

Process finished with exit code 0
```
