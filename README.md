# Guide4PathAI - A simple guide for pathology AI research
A quick start for pathology AI research by *Jiawen Li* (jw-li24@mails.tsinghua.edu.cn)

## 目录

1. [Paper](#Paper)
    - [Survey](#Survey)
    - [Popular](#Popular)
    - [Latest](#Latest)
2. [Coding](#Coding)
    - [热门](#热门)
    - [有用的project](#其他有用的project)
3. [其他资源](#其他资源)
  
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
- **Li, Bin, Yin Li, and Kevin W. Eliceiri. "Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.**
  - 简述：双流MIL模型用于WSI分类 DSMIL
  - 链接：[CVPR 2021] https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.html
- **Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer International Publishing, 2015.**
  - 简述： U-net模型（医学图像最常用的分割模型）
  - 链接：[MICCAI 2015] https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
- **Graham, Simon, et al. "Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." Medical image analysis 58 (2019): 101563.**
  - 简述： 用于细胞核语义分割的分割模型（Hover-net）
  - 链接：[MIA] https://www.sciencedirect.com/science/article/pii/S1361841519301045
- **Liu, Shuting, et al. "Unpaired stain transfer using pathology-consistent constrained generative adversarial networks." IEEE transactions on medical imaging 40.8 (2021): 1977-1989.**
  - 简述：利用非配对数据来进行HE到IHC的虚拟染色
  - 链接：[TMI] https://ieeexplore.ieee.org/abstract/document/9389763
- **Ozyoruk, Kutsev Bengisu, et al. "A deep-learning model for transforming the style of tissue images from cryosectioned to formalin-fixed and paraffin-embedded." Nature Biomedical Engineering 6.12 (2022): 1407-1419.**
  - 简述：利用生成式网络GAN将冰冻切片转成石蜡切片
  - 链接：[Nature Biomedical Engineering] https://www.nature.com/articles/s41551-022-00952-9
- **Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." Nature Medicine 30.3 (2024): 850-862.**
  - 简述：基于DINOv2的病理通用foundation model（大模型）UNI
  - 链接：[Arxiv] https://arxiv.org/abs/2308.15474; [Nature Medicine] https://www.nature.com/articles/s41591-024-02857-3
- **Huang, Zhi, et al. "A visual–language foundation model for pathology image analysis using medical twitter." Nature medicine 29.9 (2023): 2307-2316.**
  - 简述：使用twitter的病理图像-文本配对数据做CLIP预训练的大模型PLIP
  - 链接：[Nature Medicine] https://www.nature.com/articles/s41591-023-02504-3
- **Lu, Ming Y., et al. "A visual-language foundation model for computational pathology." Nature Medicine 30.3 (2024): 863-874.**
  - 简述：更大规模的病理图像-文本配对数据做CoCa预训练的大模型CONCH
  - 链接：[Arxiv] https://arxiv.org/abs/2307.12914; [Nature Medicine] https://www.nature.com/articles/s41591-023-02504-3
- **Xu, Hanwen, et al. "A whole-slide foundation model for digital pathology from real-world data." Nature (2024): 1-8.**
  - 简述：基于DINOv2的更大规模病理通用大模型 Gigapath
  - 链接：[Nature] https://www.nature.com/articles/s41586-024-07441-w

### Latest
- **Vorontsov, Eugene, et al. "A foundation model for clinical-grade computational pathology and rare cancers detection." Nature Medicine (2024): 1-12.**
  - 简述：更大规模基于DINOv2的病理通用大模型 Virchow
  - 链接: [Nature Medicine] https://www.nature.com/articles/s41591-024-03141-0 [Arxiv (v1)] https://arxiv.org/abs/2309.07778 [Arxiv (v2)] https://arxiv.org/abs/2408.00738
- **Lu, Ming Y., et al. "A Multimodal Generative AI Copilot for Human Pathology." Nature (2024): 1-3.**
  - 简述：病理ChatGPT：PathChat
  - 链接：[Nature] https://www.nature.com/articles/s41586-024-07618-3 [Arxiv] https://arxiv.org/abs/2312.07814
- **Qu, Linhao, et al. "The rise of ai language pathologists: Exploring two-level prompt learning for few-shot weakly-supervised whole slide image classification." Advances in Neural Information Processing Systems 36 (2024).**
  - 简述：利用文本prompt指导WSI分类的方法 TOP
  - 链接：[NeurIPS 2023] https://proceedings.neurips.cc/paper_files/paper/2023/hash/d599b81036fd1a3b3949b7d444f31082-Abstract-Conference.html
- **Shi J, Li C, Gong T, et al. ViLa-MIL: Dual-scale Vision-Language Multiple Instance Learning for Whole Slide Image Classification[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 11248-11258.**
  - 简述：多尺度文本指导的WSI分类方法 ViLa-MIL
  - 链接：[CVPR 2024] https://openaccess.thecvf.com/content/CVPR2024/html/Shi_ViLa-MIL_Dual-scale_Vision-Language_Multiple_Instance_Learning_for_Whole_Slide_Image_CVPR_2024_paper.html
- **Li, Jiawen, et al. "Dynamic Graph Representation with Knowledge-aware Attention for Histopathology Whole Slide Image Analysis." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.**
  - 简述：动态图神经网络的WSI分类方法 WiKG
  - 链接：[CVPR 2024] https://openaccess.thecvf.com/content/CVPR2024/html/Li_Dynamic_Graph_Representation_with_Knowledge-aware_Attention_for_Histopathology_Whole_Slide_CVPR_2024_paper.html
- **Hou, Wentai, et al. "Hybrid graph convolutional network with online masked autoencoder for robust multimodal cancer survival prediction." IEEE Transactions on Medical Imaging 42.8 (2023): 2462-2473.**
  - 简述：混合图神经网络的多模态融合策略用于生存预测分析
  - 链接：[TMI] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10061470
- **Chu, Hongbo, et al. "RetMIL: Retentive Multiple Instance Learning for Histopathological Whole Slide Image Classification." arXiv preprint arXiv:2403.10858 (2024).**
  - 简述：rention-based的序列压缩方法用于WSI分类 RetMIL
  - 链接：[Arxiv] https://arxiv.org/abs/2403.10858 [Accept by MICCAI 2024]
- **Zhang, Qilai, et al. "Leveraging Pre-trained Models for FF-to-FFPE Histopathological Image Translation." arXiv preprint arXiv:2406.18054 (2024).**
  - 简述：利用diffusion将冰冻切片转成石蜡切片
  - 链接：[Arxiv] https://arxiv.org/abs/2406.18054 [Accept by BIBM 2024] 
- **Guo, Zhengrui, et al. "HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction." arXiv preprint arXiv:2403.05396 (2024).**
  - 简述：对整张WSI进行其病理报告生成
  - 链接：[Arxiv] https://arxiv.org/abs/2403.05396 [Accept by MICCAI 2024]
- **Lu, Ming Y., et al. "Visual language pretrained multiple instance zero-shot transfer for histopathology images." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.**
  - 简述：病理大模型适配WSI的零样本策略 MI-zero
  - 链接：[CVPR 2023] https://openaccess.thecvf.com/content/CVPR2023/html/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.html
 
## Coding
### 热门
- **https://github.com/binli123/dsmil-wsi**
  - WSI的DSMIL模型库，包括WSI的处理和模型训练
- **https://github.com/mahmoodlab/CLAM**
  - WSI的CLAM模型库，包括WSI的处理和模型训练
- **https://github.com/huggingface/pytorch-image-models#introduction**
  - cv模型pytorch库

### 有用的project
- **https://github.com/lingxitong/MIL_BASELINE/tree/main**
  - MIL模型库，包括各种MIL的复现以及WSI的处理
- **https://github.com/WonderLandxD/opensdpc** 
  - .sdpc后缀的WSI文件python处理库，生强扫描仪扫出来的WSI都是sdpc格式，需要用该库进行处理（也可以处理其他通用的WSI文件）
- **More projects are waiting for your exploration**

##  其他资源
### **关于coding的平台**
1. vscode连接Ubuntu并安装Miniconda和PyTorch指南, [参考资料] https://blog.csdn.net/weixin_44795194/article/details/131248919 （或自行google/baidu，关键词：vscode，ubuntu系统，miniconda/anaconda，pytorch）
2. pycharm （配置环境麻烦，不推荐）

### **关于写作**
1. 推荐overleaf [链接] https://www.overleaf.com/project （VPN is recommend），可以部署到vscode平台中本地使用
2. 次要推荐TexPage [链接] https://www.texpage.com/?lang=zh

### **关于期刊/会议**
- [知乎：医学图像处理领域值得关注的期刊和会议](https://zhuanlan.zhihu.com/p/70225750)
- [子刊/正刊] Nature, Nature communication, Nature medicine, Nature biomedical engineering
- [人工智能会议] CVPR, ICCV, ECCV, AAAI, NeurIPS, ICML, ICLR, ACMMM
- [人工智能期刊] TPAMI, IJCV, TNNLS, TIP

### **其他有用的网站**
- https://huggingface.co/
- https://tomatocloud.me/account?action=login
- https://aideadlin.es/?sub=CV
- https://chat.deepseek.com/coder
- https://chatglm.cn/
- https://developer.aliyun.com/article/240538

### **公开数据集**
待更新...
