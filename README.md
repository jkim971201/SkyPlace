# SkyPlace - A VLSI global placer with GPU-acceleration
**Please READ**
Thanks for your attention in this repo!
I just uploaded alpha version for those who are interested in this project.
Though SkyPlace was originally coded to run bookshelf format benchmarks,
I re-implemented the whole source code to support lef/def format as well.
This made huge amount of bugs and still many of them are remaining (this is why open-sourcing has been delayed).
Please do not expect this code will run safely.
Especially, bookshelf flow is currently not available (you can run by read_bookshelf command but the results are not valid).
I will upload more reliable version as soon as possible.

# Publication
- Jaekyung Im and Seokhyeong Kang,
  "SkyPlace : A New Mixed-size Placement Framework using Modularlity-based Clustering and SDP Relaxation", ACM/IEEE Design Automation Conference (DAC), 2024.

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
./SkyPlace ../test/test_lefdef.tcl
```

# Acknowledgement
Many thanks to [DREAMPlace](https://github.com/limbo018/DREAMPlace) which has inspired this project.
