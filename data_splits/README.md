

# New splits for public datasets used in technical report

This repo contains new splits for the VQA-Rad and PAD-UFES-20 datasets.

If you use any of these in your own research, please cite our technical report:

```
@misc{yang2024advancing,
      title={Advancing Multimodal Medical Capabilities of Gemini}, 
      author={Lin Yang and Shawn Xu and Andrew Sellergren and Timo Kohlberger and Yuchen Zhou and Ira Ktena and Atilla Kiraly and Faruk Ahmed and Farhad Hormozdiari and Tiam Jaroensri and Eric Wang and Ellery Wulczyn and Fayaz Jamil and Theo Guidroz and Chuck Lau and Siyuan Qiao and Yun Liu and Akshay Goel and Kendall Park and Arnav Agharwal and Nick George and Yang Wang and Ryutaro Tanno and David G. T. Barrett and Wei-Hung Weng and S. Sara Mahdavi and Khaled Saab and Tao Tu and Sreenivasa Raju Kalidindi and Mozziyar Etemadi and Jorge Cuadros and Gregory Sorensen and Yossi Matias and Katherine Chou and Greg Corrado and Joelle Barral and Shravya Shetty and David Fleet and S. M. Ali Eslami and Daniel Tse and Shruthi Prabhakara and Cory McLean and Dave Steiner and Rory Pilgrim and Christopher Kelly and Shekoofeh Azizi and Daniel Golden},
      year={2024},
      eprint={2405.03162},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


##VQA-Rad: Balanced splits and exclusions from human evaluations

File: **vqa_rad_balanced_split_and_human_eval_inclusions.tsv** 

This file contains new balanced three-way splits (train, validate, test) for the VQA-Rad dataset (column BALANCED_SPLIT). 

In addition, column INCLUDED_IN_HUMAN_EVAL indicates which question and answers were excluded in the human evaluation of the test split in our Technical Report [Advancing Multimodal Medical Capabilities of Gemini](https://arxiv.org/abs/2405.03162). See the Appendix of the report for details on the motivation and criteria for the new splits.


The original VQA-RAD dataset was published by:

Dina Demner-Fushman (ddemner@mail.nih.gov)\
Lister Hill National Center for Biomedical Communications, National Library of Medicine, Bethesda, MD, USA,

hosted at https://osf.io/89kps/.


##PAD-UFES-20: Train and test split

File: **pad_ufes_20_split.tsv** 

This file contains a split for the PAD-UFES-20 dataset into a train and test subset, which was used to train and test the Med-Gemini model described in [Advancing Multimodal Medical Capabilities of Gemini](https://arxiv.org/abs/2405.03162). The split is at the patient level, i.e. patients are disjoint between the two splits.
Column PATIENT_ID contains the patient ID, column SPLIT the subset assignment.

The original PAD-UFES-20 dataset was published by:

Pacheco, Andre G. C.; Lima, Gustavo R.; Salomão, Amanda S.; Krohling, Breno; Biral, Igor P.; de Angelo, Gabriel G. ; Alves Jr, Fábio  C. R. ; Esgario, José G. M.; Simora, Alana C. ; Castro, Pedro B. C. ; Rodrigues, Felipe B.; Frasson, Patricia H. L. ; Krohling, Renato A.; Knidel, Helder ; Santos, Maria C. S. ; Espírito Santo, Rachel B.; Macedo, Telma L. S. G.; Canuto, Tania R. P. ; de Barros, Luíz F. S. (2020), “PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones”, Mendeley Data, V1, doi: 10.17632/zr7vgbcyr2.1

and is hosted at https://data.mendeley.com/datasets/zr7vgbcyr2/1.