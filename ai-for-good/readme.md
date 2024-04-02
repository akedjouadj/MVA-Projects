In this project, we aimed to **estimate the energy consumption of a simple roberta-base** mathematics exercices classifier.

- **About the model**

We trained a mathematics exercices classifier using a roberta-base language model of 1 million parameters. The model was initialized from scratch and pretrained on a corpus of ~75k tokens with a vocabulary size of ~3.8K tokens.

- **On the ridiculous energy cost of state-of-the-art performance**

**42.2%** of the **energy consumed** during the **pretraining** of the model was found to be **unnecessary** for achieving good performance on the final task. Similarly, **57.1%** of the **energy consumed** during the model’s **finetuning** on the task of interest was deemed unnecessary. These proportions are <span style="color:red">**extremely high**</span>, and such experiments conducted at a very small scale (only 1M parameters) send strong signals about the need to adopt **energy consumption as a de facto metric** in the training of such systems to achieve SOTA performance, **especially in the era of foundation models**.

- **Illustration of the energy consumption as an exponential function of performance metrics**

![pretraining energy consumed against crossentropy losses](images/pretraining%20energy%20consumed%20against%20crossentropy%20losses.png)


![energy conso of finetunings against accuracy](images/energy%20conso%20of%20finetunings%20against%20accuracy.png)

