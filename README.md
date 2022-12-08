# Extracting Social Determinants of Health from Electronic Health Records using Natural Language Processing

Currently, meaningful clinical data is lost within unstructured sequential data in electronic health records (EHRs). We propose training natural language processing (NLP) models on EHRs containing doctors’ notes of their diagnosis and patients' conversations to uncover hidden word associations with social determinants of health (SDoH), and specifically food insecurity.

# Dataset

We harnessed data in the form of electronic health records (EHRs) from the Stanford Medicine Research Data Repository (STARR). This data has been collected from the Adult Hospital and the Lucile Packard Children’s Hospital under IRB approval. The data includes information on names, dates, medical record numbers, account numbers, demographic identifiers, lab/test results, diagnosis or procedure codes, clinical narratives/reports, prescriptions or medications, and imaging reports.

Our dataset has been manually labeled with a true or false indication for food insecurity, the social determinant of health we’ve chosen to focus on in this project. The dataset was labeled with the help of Akshay Swaminathan in the Stanford School of Medicine. In terms of the breakdown of the demographics of our dataset, we have a total of 259 clinical patient notes. Of those 259, 120 were male patients, 131 were female, and 8 were unclassified/other. Additionally, 84 patient notes were labeled true for the existence of food insecurity, while 175 were labeled false.

# Methodology

We decided to establish a baseline of performance prior to attempting in-context learning to understand how a pre-trained BERT model would perform when simply fine-tuned to our data and hyperparameters. To find the best hyperparameters for our model, we used a grid search to methodically evaluate each model based on varied hyperparameters. We first split our data into a training, validation, and test set to evaluate the model on our data. We then tokenized and encoded the sentence sequences using the BERT tokenizer, padding to a maximum sentence length of 100, and truncating past the maximum for extended sequences. In our forward propagation, we used a leaky relu activation and a softmax activation function but also froze the preexisting parameters while running our model. Finally, we used an Adam optimizer with a learning rate of 0.00001 and a binary cross entropy loss function. After running BERT on our electronic health record data without in context learning, we established that the model achieved an accuracy of about 65%. The precision for false predictions was 0.69 with a recall of 0.86 and an F1-score of 0.77. On the other hand, the precision for true predictions was 0.45 with a recall of 0.24 and an F1-score of 0.31. This acted as our baseline model to compare our in-context learning integration model with.

In order to improve upon our baseline of a pre-trained BERT model, we decided to incorporate an emergent property of language models that has not yet been fully explored: in-context learning. To provide a more robust definition of in-context learning, it is essentially a Bayesian framework for “locating” latent concepts acquired by a language model by conditioning the LM simply on an input-output example, rather than optimizing hyperparameters using backpropagation. An input-output example would thus consist of the pairing of a question-answer, such as “Q: Does this person have food insecurity?” and “A: 1” or “A: 0” depending on the true label in the data. In-context learning prompts are a list of concatenated IID (independent and identically distributed) training examples. The diagram below illustrates how this would work as the input to our model.

<img src="/Users/virajmehta/Desktop/ICL_Diagram.PNG" alt="Alt text" title="In-Context Learning Diagram">

# References

Special thanks to Akshay Swaminathan in the Stanford School of Medicine for his guidance and mentorship, as well as his provision of the labeled STARR dataset.

“Bert 101 - State of the Art NLP Model Explained.” BERT 101 - State of the Art NLP Model
Explained, https://huggingface.co/blog/bert-101#2-how-does-bert-work.

Khurshid, Shaan, et al. “Cohort Design and Natural Language Processing to Reduce Bias in
Electronic Health Records Research.” Nature News, Nature Publishing Group, 8 Apr.
2022, https://www.nature.com/articles/s41746-022-00590-0#Sec8.

Lin, J., Jiao, T., Biskupiak, J. E., & McAdam-Marx, C. (2013). Application of electronic medical
record data for health outcomes research: a review of recent literature. Expert review of
pharmacoeconomics & outcomes research, 13(2), 191–200. https://doi.org/10.1586/
erp.13.7.

“Natural language processing in healthcare medical records.” ForeSee Medical. (n.d.). Retrieved 
October 13, 2022, https://www.foreseemed.com/natural-language-processing-in-
healthcare.

Patra, Braja et al. “Extracting Social Determinants of Health from Electronic Health Records
Using Natural Language Processing: A Systematic Review.” Journal of the American
Medical Informatics Association: JAMIA, U.S. National Library of Medicine,
https://pubmed.ncbi.nlm.nih.gov/34613399/.

Ruan, Xiaowen et al. (2021, March 2). “Health-adjusted life expectancy (Hale) in Chongqing,
China, 2017: An Artificial Intelligence and big data method estimating the burden of disease at City Level”. The Lancet Regional Health - Western Pacific. 2 March 2021, https://www.sciencedirect.com/science/article/pii/S2666606521000195?pes=vor.

“Tapping into EHR text fields with NLP.” Prometheus Research Data Management Solutions.
11 July 2022,
https://www.prometheusresearch.com/using-natural-language-processing-nlp-to-make-us-of-ehr-text-fields-for-medical-research/#:~:text=Previous%20Next-,Using%20Natural%20Language%20Processing%20(NLP)%20to%20Make%20Use%20of%20Electronic,data%20on%20a%20massive%20scale.

Xie, Sang Michael and Sewon Min. “How Does In-Context Learning Work? A Framework for
 	Understanding the Differences from Traditional Supervised Learning.” SAIL Blog, 1
Aug. 2022, https://ai.stanford.edu/blog/understanding-incontext/.

Xie, Sang Michael et al. “An Explanation of In-Context Learning as Implicit Bayesian
Inference.” Cornell University, 21 July 2022, https://arxiv.org/abs/2111.02080. 

Ye, Xi and Greg Durrett. “The Unreliability of Explanations in Few-shot Prompting for Textual
Reasoning.” 36th Conference on Neural Information Processing Systems, 2022,
https://arxiv.org/pdf/2205.03401.pdf.
