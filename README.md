# Guide4PathAI - A simple guide for pathology AI research
A quick start for pathology AI research by *Jiawen Li* (jw-li24@mails.tsinghua.edu.cn)

## 目录

1. [Paper](#Paper)
    - [Survey](#Survey)
    - [Popular](#Popular)
    - [Latest](#Latest)
2. [Coding](#Coding)
    - [权威性高的project](#权威性高的project)
    - [其他有用的project](#其他有用的project)
3. [其他资源](#其他资源)
    - [数据集](#数据集)
    - [工具和库](#工具和库)
    - [社区和论坛](#社区和论坛)
  
## Paper
### Survey
- **Song, Andrew H., et al. "Artificial intelligence for digital and computational pathology." Nature Reviews Bioengineering 1.12 (2023): 930-949.**
  - 简述：2023年病理AI/数字病理/计算病理综述（偏算法）
  - 链接：[Arxiv] https://arxiv.org/abs/2401.06148; [Nature Reviews Bioengineering] https://arxiv.org/abs/2401.06148
- **Hosseini, Mahdi S., et al. "Computational pathology: a survey review and the way forward." Journal of Pathology Informatics (2024): 100357.**
  - 简述：2024年病理AI/计算病理综述（偏应用）
  - 链接: [Journal of Pathology Informatics] https://www.sciencedirect.com/science/article/pii/S2153353923001712
- **Lipkova, Jana, et al. "Artificial intelligence for multimodal data integration in oncology." Cancer cell 40.10 (2022): 1095-1110.**
  - 简述：2022年肿瘤学人工智能的多模态学习综述
  - 链接：[Cancer cell] https://www.cell.com/cancer-cell/fulltext/S1535-6108(22)00441-X

### Popular
- **Lu, Ming Y., et al. "Data-efficient and weakly supervised computational pathology on whole-slide images." Nature biomedical engineering 5.6 (2021): 555-570.**
  - 简述：无需像素级标注的一种弱监督学习的病理WSI分类方法（CLAM）
  - 链接：[Arxiv] https://arxiv.org/abs/2004.09666; [Nature biomedical engineering] https://www.nature.com/articles/s41551-020-00682-w
- **Lu, Ming Y., et al. "AI-based pathology predicts origins for cancers of unknown primary." Nature 594.7861 (2021): 106-110.**
  - 简述：利用Attention-based multiple instance learning （ABMIL）对未知原发癌的起源做预测（slide-level的多分类任务）
  - 链接：[Nature] https://www.nature.com/articles/s41586-021-03512-4
- **Ilse, Maximilian, Jakub Tomczak, and Max Welling. "Attention-based deep multiple instance learning." International conference on machine learning. PMLR, 2018.**
  - 简述：Attention-based MIL （ABMIL）的方法介绍
  - 链接：[ICML 2018] https://proceedings.mlr.press/v80/ilse18a.html?ref=https://githubhelp.com
- **Chen, Richard J., et al. "Scaling vision transformers to gigapixel images via hierarchical self-supervised learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.**
  - 简述： 针对WSI量身定做的Vision Transformer模型 (HIPT)
  - 链接：[CVPR 2022] https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Scaling_Vision_Transformers_to_Gigapixel_Images_via_Hierarchical_Self-Supervised_Learning_CVPR_2022_paper.html?trk=public_post_comment-text
- **Shao, Zhuchen, et al. "Transmil: Transformer based correlated multiple instance learning for whole slide image classification." Advances in neural information processing systems 34 (2021): 2136-2147.**
  - 简述： 基于Transformer的类MIL模型 （TransMIL）
  - 链接：[NeurIPS 2021] https://proceedings.neurips.cc/paper_files/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html
- **Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer International Publishing, 2015.**
  - 简述： U-net模型（医学图像最常用的分割模型）
  - 链接：[MICCAI 2015] https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
- **Graham, Simon, et al. "Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." Medical image analysis 58 (2019): 101563.**
  - 简述： 用于细胞核语义分割的分割模型（Hover-net）
  - 链接：[MIA] https://www.sciencedirect.com/science/article/pii/S1361841519301045
- **Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." Nature Medicine 30.3 (2024): 850-862.**
  - 简述：基于自监督学习的病理通用foundation model（大模型）UNI
  - 链接：[Arxiv] https://arxiv.org/abs/2308.15474; [Nature Medicine] https://www.nature.com/articles/s41591-024-02857-3
- **Huang, Zhi, et al. "A visual–language foundation model for pathology image analysis using medical twitter." Nature medicine 29.9 (2023): 2307-2316.**
    - 

