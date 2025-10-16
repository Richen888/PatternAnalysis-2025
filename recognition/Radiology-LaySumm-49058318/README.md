# Project: 13 - Fine-tune FLAN-T5 for Radiology Report Simplification (hard difficulty)
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
Algorithem picture:
<img width="1089" height="204" alt="image" src="https://github.com/user-attachments/assets/68e9c4fc-4b1e-43b7-b689-1cef2a621035" />
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
#### Error Analysis
Despite strong ROUGE scores, the model occasionally misplaces anatomical locations (e.g., "lung apex" vs. "lung base") and may oversimplify complex pathological descriptions. These errors likely stem from limited anatomical context in the training data and the challenge of balancing accuracy and simplicity.
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
## 7.Training Information
### 7.1 Hardware Configuration
GPU: NVIDIA RTX5060 laptop  
VRAM: 8GB  
CPU: 13th Gen Intel(R) Core(TM) i7-13700HX  
RAM: 16X2 32GB 4800MT/S DDR5  
Storage: 1TB    
#### Configuration
<img width="998" height="277" alt="image" src="https://github.com/user-attachments/assets/f21f3b73-3aec-4adb-9507-ff75f985b1cd" />  

Model name: google/flan-t5-base    
LoRA rank (r): 16    
LoRA Alpha: 32    
LoRA Dropout: 0.1    
Target modules: ["q", "v"]    
Total parameter quantity: 249,347,328    
Number of trainable parameters: 1,769,472    
Parameter training ratio: 0.71%    
Number of training rounds: 10 epoch    
Batch size: 4    
Learning rate: 5e-5    
Weight decay: 0.01    
Preheating steps: 10    
Gradient clipping: 0.5    
Input length: 512 tokens       
Output length: 256 tokens    
train_runtime: 57782.5212(16h), train_samples_per_second: 26.038, train_steps_per_second: 6.51, train_loss: 0.8983025245932217(from 2.96 to 0.7), epoch: 10.0
## 8. Usage Instructions
### 8.1 Environment
Python 3.11.13    
PyTorch: 2.9.0.dev20250810+cu128    
CUDA: 12.8   
#### Install dependencies
torch>=2.0.0    
transformers>=4.30.0    
datasets>=2.12.0    
accelerate>=0.20.0    
peft>=0.4.0    
bitsandbytes>=0.40.0    
rouge-score>=0.1.2    
nltk>=3.8.0    
numpy>=1.24.0    
pandas>=1.5.0    
tqdm>=4.64.0    
       pip install -r requirements.txt
### 8.2 Train the Model
       python train.py
### 8.3 Run Inference
After training, run the following to verify ROUGE scores:  
       python predict.py
### 8.4 Explanation of Training Technology Selection 
Taking into account the actual requirements of the project and the limitations of computing resources, we have made the following technical choices:        
#### Single GPU Training      
- Reasons for Selection: The project uses NVIDIA RTX 5060 (8GB VRAM), and a single GPU is sufficient to meet the training requirements.  
- Actual Results: It took 16 hours to complete 10 epochs of training, and the validation loss decreased from 2.96 to 0.7, demonstrating significant training effectiveness.     
#### FP32 Precision Training      
- Reason for Selection: To ensure training stability and avoid the numerical precision loss that may occur with mixed precision  
- Actual Verification: The training process converged stably without any gradient explosion or NaN issues   
#### No Gradient Accumulation     
- Reason for Selection: With a batch size of 4, the 8GB VRAM has achieved the optimal utilization rate, and no further accumulation is necessary.  
- Resource Utilization: The GPU utilization rate remains between 85% and 95%, and the resources are fully utilized.     
#### Technical Feasibility Explanation  
Although the current implementation uses a single GPU for training, the code architecture supports expansion to distributed training:  
- Utilizing the Hugging Face Transformers library, which naturally supports multi-GPU parallelism  
- The data loader design supports distributed data parallelism  
- To scale up to a larger scale, simply configure to enable `torchrun` distributed training

## 9.References
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
