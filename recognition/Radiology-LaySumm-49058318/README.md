# Project - Generation of simplified summaries of radiology reports
## 1.Project Objectives
This project aims to develop an automated radiology report simplification system based on deep learning techniques, which converts specialized medical imaging diagnostic reports into easily understandable layperson summaries. The system utilizes the FLAN-T5 pre-trained language model combined with LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning, specifically trained on the BioLaySumm 2025 dataset to achieve intelligent transformation from professional medical terminology to everyday language.

#### Core Objectives Include:

1.Addressing Information Asymmetry in Healthcare: Radiology reports typically contain numerous technical terms and anatomical descriptions (e.g., "heterogeneously enhancing mass with peripheral washout") that are difficult for general patients to comprehend. This project employs natural language generation technology to translate these complex contents into plain language understandable by non-specialists, thereby helping patients better grasp their health conditions.

2.Improving Accessibility of Medical Information: In resource-constrained healthcare environments, physicians often lack sufficient time to explain report details to every patient. This system can serve as an auxiliary tool, automatically generating easy-to-understand report summaries, alleviating the workload of healthcare professionals while ensuring patients receive accurate medical information.

3.Exploring Text Simplification Techniques in Specialized Domains: Radiology reports possess specific linguistic structures and specialized vocabulary systems. This project investigates how to perform effective transfer learning on such highly specialized texts, providing technical references for text simplification in other professional domains (e.g., legal documents, scientific papers).

4.Implementing Parameter-Efficient Fine-Tuning Strategies: By employing LoRA technology, the project significantly reduces the number of trainable parameters (from approximately 250 million parameters to only about 2.5% being trained) while maintaining model performance, exploring the feasibility of customizing domain-specific models under limited computational resources.

This project's technical solution directly addresses the practical needs of the ACL 2025 BioLaySumm Workshop, providing a comprehensive end-to-end solution for practical applications in medical natural language processing, including complete pipeline implementation from data preprocessing and model training to performance evaluation and deployment inference.
## 2. Algorithm Principles
### 2.1 Model Architecture

This project is based on the FLAN-T5 (Fine-tuned LAnguage Net - Text-to-Text Transfer Transformer) architecture, a sequence-to-sequence model specifically optimized for instruction-following tasks. Building upon the original T5 model, FLAN-T5 significantly enhances instruction-following capabilities and zero-shot generalization through multi-task fine-tuning on a wide range of instruction-based tasks.

#### Core Architectural Components:

1.Encoder-Decoder Structure: Employs the standard Transformer encoder-decoder architecture, where the encoder processes input expert radiology reports and the decoder generates simplified layperson summaries.

2.Relative Position Encoding: Utilizes T5's relative position bias mechanism instead of traditional absolute position encoding, providing better handling of long text sequences.

3.Text-to-Text Unified Framework: Unifies all natural language processing tasks into a text-to-text transformation format, with inputs formatted as "Simplify the following medical report: [original text]" and outputs as simplified layperson versions.

4.Prefix Language Modeling: Prepends task-specific prefix text to encoder inputs, guiding the model to perform the specific text simplification task.

#### LoRA Fine-tuning Mechanism:  
This project employs LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning, based on the mathematical principle of low-rank matrix decomposition: 

            ΔW = BA  
where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r ≪ min(d,k), injecting trainable adapters into the Query and Value projection layers of the Transformer, significantly reducing the number of parameters requiring training.

#### Forward Propagation Formula:

After injecting LoRA adapters, the forward propagation calculation becomes:  

            h = W₀x + ΔWx = W₀x + BAx

The meaning of this formula is:

W₀x: Forward computation of the original pre-trained model, preserving its existing knowledge capabilities  
BAx: Incremental update from LoRA adapters, specifically learning task-specific knowledge  
Addition operation: Combines original capabilities with task-specific knowledge for efficient transfer learning

#### Advantage Analysis:

The advantages of this design include:

1.Parameter Efficiency: Only requires training low-rank matrices A and B, significantly reducing the number of trainable parameters

2.Knowledge Preservation: Keeps original weights W₀ unchanged, avoiding catastrophic forgetting

3.Flexible Deployment: BA can be merged back into W₀ during inference, without adding inference latency

By injecting these trainable adapters into the Query and Value projection layers of the Transformer, the number of parameters requiring training is reduced from 100% to only about 2.5%.

### 2.2 Training Strategy

#### Training Objective Function:

The model learns the mapping from expert reports to layperson summaries by maximizing conditional likelihood:  

<img width="351" height="90" alt="屏幕截图 2025-10-14 235944" src="https://github.com/user-attachments/assets/8c46a00b-0a1b-4e88-b87f-31d01293269d" />


where x represents the input expert report and y represents the target layperson summary.

#### Multi-stage Training Pipeline:

1.Instruction-aware Pre-training: Leverages FLAN-T5's pre-trained prior knowledge from multiple instruction-based tasks

2.Domain Adaptation Fine-tuning: Conducts domain-specific adaptive training on medical text data

3.Task Specialization Training: Performs end-to-end training specifically for the radiology report simplification task

#### Optimization Methodology:

Employs Teacher Forcing training strategy, using ground truth prefix tokens to predict the next token

Uses cross-entropy loss function to optimize sequence generation tasks

Implements gradient accumulation and gradient clipping to stabilize the training process
## 3.Project File Structure
### 3.1 Core Code Files
Radiology-LaySumm-49058318/  
├── modules.py           # Model definition and LoRA configuration  
├── dataset.py           # Data loading and preprocessing pipeline  
├── train.py             # Training loop and model training logic  
├── predict.py           # Inference script and result generation  
├── utils.py             # Helper functions and utility classes  
├── requirements.txt     # Python dependency environment configuration  
└── README.md           # Project documentation and usage instructions  
### 3.2 Auxiliary Tool Files
#### Configuration Files:  
requirements.txt - List of Python package dependencies required for project execution  
training.log - Automatically generated log file during training process  
#### Output Directories:
radiology-simplifier-output/ - Trained model weights and checkpoints  
biolaysumm_validation_results.csv - Validation set inference results storage file  
## 4. Dataset Usage Instructions
### 4.1 Dataset Source  
This project utilizes the BioLaySumm 2025 dataset from the ACL 2025 BioLaySumm Workshop open track. This dataset is specifically designed for medical report simplification tasks, containing expert radiology reports paired with corresponding human-written layperson summaries.
### 4.2 Data Format and Structure
The dataset follows twe-split structure:  
Training Set: Primary data for model parameter training  
Validation Set: Reserved for final model performance evaluation  
(Since the official training set has no standard answers, it cannot be used to evaluate the ROUGE score. Therefore, the validation set is used for evaluation)
#### Data Field Descriptions:
radiology_report: Original expert radiology report (input text)  
layman_report: Simplified layperson version summary (target text)  
Other metadata fields: Report ID, source institution, and other auxiliary information  
### 4.3 Data Preprocessing Pipeline
#### Text Cleaning Steps:  
1.Remove special characters and excess whitespace  
2.Standardize text encoding format (UTF-8)  
3.Handle consistency of medical abbreviations and terminology  
Input Formatting:  
Wrap original reports in instruction format:

"Simplify the following medical report: [radiology_report]"

#### Label processing:  
Use "layman_report" directly as the training target to ensure that the model learns to map from professional terms to plain language expressions.
## 5. Example Inputs and Outputs
Example 1:  
Radiology Report: The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax.  
Reference Lay Summary: The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs.    
Generated Summary: The chest x-ray shows a lot of trapped air in the lungs. There are long-term changes at the top of both lungs. The upper back is curved more than normal. There is no sign of air leakage outside the lungs.    

--------------------------------------------------------------------------------    
Example 2:  
Radiology Report: Central venous catheter traversing the left jugular vein with its tip in the superior vena cava. The remainder is unchanged.    
Reference Lay Summary: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else is the same as before.    
Generated Summary: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else looks the same as before.    

--------------------------------------------------------------------------------      
Example 3:    
Radiology Report: Chronic pulmonary changes      
Reference Lay Summary: Long-term changes in the lungs are seen.      
Generated Summary: Long-term changes in the lungs are seen.      

--------------------------------------------------------------------------------    
Example 4:    
Radiology Report: Radiological signs of air trapping, flattened diaphragm, and increased retrosternal space. Calcified pleural plaques at the level of the left diaphragmatic pleura. Loss of volume in the left lung with subpleural linear opacities. Findings are related to chronic inflammatory changes due to asbestos e...    
Reference Lay Summary: The X-ray shows signs of trapped air, a flattened muscle under the lungs, and more space behind the breastbone. There are also hardened areas on the lung lining on the left side. The left lung has lost some volume and has some linear shadows near the outer lining. These findings are related to long-term inflammation caused by exposure to asbestos.  Looking at the previous CT scan, there are no significant changes compared to the scanogram dated 3/4/2009.    
Generated Summary: The x-ray shows signs of trapped air in the lungs, a flattened diaphragm, and increased space behind the breastbone. There are also calcified plaques on the left side of the diaphragm's lining. The left lung has less volume, and there are linear opacities near the lung surface. These findings are related to long-term inflammation due to asbestos exposure. The previous CT scan shows no significant changes compared to the scanogram dated 3/4/2009.    

--------------------------------------------------------------------------------  
Example 5:    
Radiology Report: Calcified granuloma in the right lung vertex.    
Reference Lay Summary: There is a calcified granuloma located at the top of the right lung.    
Generated Summary: There is a calcified granuloma, which is a type of hardened lump, in the lower part of the right lung.  

--------------------------------------------------------------------------------  
## 6. Validation Set ROUGE Scores and Interpretation
#### ROUGE Evaluation Results
✅ ROUGE Scores on Validation Set:  
ROUGE-1: 0.7316  
ROUGE-2: 0.5510  
ROUGE-L: 0.6826    
ROUGE-Lsum: 0.6825    
#### In-depth Score Interpretation
ROUGE-1 (0.7316) - Excellent  
Indicates outstanding performance at the word-level recall  
Accurately captures and retains key medical terminology and important vocabulary from original reports  
0.73 unigram overlap rate demonstrates high consistency between generated content and reference summaries  

ROUGE-2 (0.5510) - Good  
Reflects model's capability in phrase and word-pair level matching  
0.5510 bigram overlap rate shows the model can generate coherent medical expressions  
The decrease compared to ROUGE-1 is normal, as word-pair matching is more challenging than word-level matching  

ROUGE-L (0.6826) and ROUGE-Lsum (0.6825) - Excellent  
Demonstrates excellent performance at the long-sequence and sentence structure level  
Maintains logical order and information organization structure of the original text  
Nearly 0.7 longest common subsequence matching rate proves high structural similarity between generated text and reference summaries  

#### Comprehensive Performance Analysis
Strengths Demonstrated:    
High-accuracy terminology conversion - High ROUGE-1 score proves accurate medical term translation  
Good coherence - ROUGE-2 indicates naturally fluent generated sentences  
Structural integrity - High ROUGE-L series scores demonstrate reasonable information organization structure  
#### Areas for Improvement:
1. Medical Entity Recognition Accuracy
Main Issue: Incorrect anatomical location descriptions (e.g., "lung apex" misidentified as "lung base")
Solutions:  
Integrate professional medical NER model for preprocessing  
Establish anatomical terminology verification rule base  
Add specialized training data for spatial relationships  

2. Complex Phrase Generation Optimization  
Main Issue: Relatively low ROUGE-2 score, phrase coherence needs improvement  
Solutions:  
Introduce phrase-level contrastive learning training  
Enhance learning of common medical expression patterns  
Optimize beam search parameters for better generation fluency  

3. Clinical Safety Mechanism Enhancement  
Main Issue: Potential risk of medical misinformation propagation  
Solutions:  
Establish key medical assertion verification mechanism  
Implement uncertainty expression preservation function  
Develop automatic detection and alerting for critical errors  

4. Inference Efficiency Optimization  
Main Issue: 2.08 seconds/sample inference speed limits clinical application  
Solutions:  
Apply model quantization and pruning techniques  
Optimize batch processing and caching mechanisms  
Explore hardware acceleration solutions

## 10.References
1. T5 Model  
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020).  
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.  
Journal of Machine Learning Research, 21(140), 1-67.  
Paper: https://jmlr.org/papers/v21/20-074.html  
arXiv: https://arxiv.org/abs/1910.10683  
2. FLAN-T5 Model      
Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., et al. (2022).  
Scaling Instruction-Finetuned Language Models.  
arXiv preprint arXiv:2210.11416.  
arXiv: https://arxiv.org/abs/2210.11416  
3. LoRA Method  
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).  
LoRA: Low-Rank Adaptation of Large Language Models.  
International Conference on Learning Representations (ICLR).  
Paper: https://openreview.net/forum?id=nZeVKeeFYf9  
arXiv: https://arxiv.org/abs/2106.09685   
4. Hugging Face Transformers  
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., et al. (2020).  
Transformers: State-of-the-Art Natural Language Processing.  
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38-45.  
Paper: https://www.aclweb.org/anthology/2020.emnlp-demos.6/  
GitHub: https://github.com/huggingface/transformers  
5. PEFT Library  
Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., & Paul, S. (2022).  
PEFT: State-of-the-Art Parameter-Efficient Fine-Tuning Methods.   
GitHub: https://github.com/huggingface/peft  
6. Accelerate Library  
Gugger, S., Debut, L., Wolf, T., Schmid, P., Mueller, Z., Mangrulkar, S., Sun, M., & Bossan, B. (2022).  
Accelerate: Training and inference at scale.  
GitHub Repository: https://github.com/huggingface/accelerate  
Documentation: https://huggingface.co/docs/accelerate/index  
7. ROUGE Evaluation Metric  
Lin, C.-Y. (2004).  
ROUGE: A Package for Automatic Evaluation of Summaries.  
In Proceedings of the Workshop on Text Summarization Branches Out, 74-81.  
Paper: https://aclanthology.org/W04-1013/  
8. bitsandbytes Library  
Dettmers, T. (2021).  
*8-bit Optimizers via Block-wise Quantization.*  
GitHub Repository: https://github.com/TimDettmers/bitsandbytes  
Documentation: https://huggingface.co/docs/bitsandbytes/index  
9. BioLaySumm Dataset  
BioNLP Workshop. (2025).  
BioLaySumm 2025 Shared Task: Layman Summarization of Radiology Reports.  
Proceedings of the BioNLP Workshop.  
Dataset: https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track  
10. Sequence-to-Sequence Learning  
Sutskever, I., Vinyals, O., & Le, Q. V. (2014).  
Sequence to Sequence Learning with Neural Networks.  
Advances in Neural Information Processing Systems, 27.  
Paper: https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf  
arXiv: https://arxiv.org/abs/1409.3215  
11. Attention Mechanism  
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).  
Attention Is All You Need.  
Advances in Neural Information Processing Systems, 30.  
Paper: https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf  
arXiv: https://arxiv.org/abs/1706.03762  
12. Medical Text Simplification  
Devaraj, A., Marshall, I. J., Wallace, B. C., & Li, J. J. (2021).  
Paragraph-level Simplification of Medical Texts.  
Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics.  
Paper: https://aclanthology.org/2021.naacl-main.371/  
13. Transfer Learning in NLP  
Ruder, S., Peters, M. E., Swayamdipta, S., & Wolf, T. (2019).  
Transfer Learning in Natural Language Processing.  
Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Tutorials.  
Paper: https://aclanthology.org/N19-5004/  
