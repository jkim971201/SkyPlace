# SkyPlace - A VLSI global placer with GPU-acceleration
**Source code will be uploaded soon.**
I'm currently working for bug fixing. Sorry to people who have been waiting for this repo. 

# Publications
- Jaekyung Im and Seokhyeong Kang,
  "**SkyPlace : A New Mixed-size Placement Framework using Modularlity-based Clustering and SDP Relaxation**", ACM/IEEE Design Automation Conference (DAC), San Francisco, CA, June 27-31, 2024

# Dependency
- GCC
  - Tested on GCC 8/9/10/11
- CUDA
  - Tested on 11.8
- Qt
  - Tested on Qt5
- Mosek
  - You need Mosek C++ fusion API as a semidefinite programming solver.
  - You can get free license for academic use [here](https://www.mosek.com/products/academic-licenses).
- Eigen3
- X11
- Jpeg 
- Flex
- Bison
- TCL 8.6

# How to build
```
git clone https://github.com/jkim971201/SkyPlace
cd SkyPlace
mkdir build & cd build
cmake .. -DMOSEK_INSTALL="${mosek_install_path}"
make -j
```

# How to run
```
./SkyPlace test_lefdef.tcl
```
