# Utilities for sorting and annotating Computed Tomography DICOMs.


This repo contains utility code for reading DICOMs from a Google Cloud DICOM
store, sorting axial CTs to prepare Tensorflow examples, and for annotating
DICOMs from model results, as described in "Assistive AI in Lung Cancer Screening: A Retrospective Multinational Study in the United States and Japan" (https://doi.org/10.1148/ryai.230079).



If you use this software in your own research, please cite our paper:

```
@article{doi:10.1148/ryai.230079,
author = {Kiraly, Atilla P. and Cunningham, Corbin A. and Najafi, Ryan and Nabulsi, Zaid and Yang, Jie and Lau, Charles and Ledsam, Joseph R. and Ye, Wenxing and Ardila, Diego and McKinney, ScottM. and Pilgrim, Rory and Liu, Yun and Saito, Hiroaki and Shimamura, Yasuteru and Etemadi, Mozziyar and Melnick, David and Jansen, Sunny and Corrado, Greg  S. and Peng, Lily and Tse, Daniel and Shetty, Shravya and Prabhakara, Shruthi and Naidich, David P. and Beladia, Neeral and Eswaran, Krish},
title = {Assistive AI in Lung Cancer Screening: A Retrospective Multinational Study in the United States and Japan},
journal = {Radiology: Artificial Intelligence},
volume = {0},
number = {ja},
pages = {e230079},
year = {0},
doi = {10.1148/ryai.230079},
    note ={PMID: 38477661},
URL = {https://doi.org/10.1148/ryai.230079
},
eprint = {https://doi.org/10.1148/ryai.230079
}
}
```