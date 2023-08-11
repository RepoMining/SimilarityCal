# SimilarityCal

There are two strategies to calculate the similarity between two repositories based on repo embedding.

Finally, two bi-classification based on repo embedding are selected, and the models have been kept in ```TWINS_MODEL``` directories.

All saved models has similar design including a shared feature selector and a classifier.

![image.png](./assets/image.png)


## 1. Mean strategy

In ```mean``` directory, the code in notebook show all works for mean strategy.

We are aware that identical sub-embedding from different repositories may differ. In order to enable their computation within a neural network, it is necessary to standardize their lengths. If each type of embedding is one feature, there are five features. The “mean” strategy is that because each feature has different length, the embedding will be calculated mean value in the first dimension. As a result, all embedding will be compressed as a 1 × 768 matrix.


**Multi-classification:**

The idea of the multi-classification is to observe the accuracy of the model. If the accuracy is high and available, the output of the *softmax* layer which is a vector consisting of 130 probabilities can be used to calculate the cosine similarity.

In this part, each single embedding has been tested and the repo embbeding also is trained. The results are not good, it is overfitting due to limit of dataset.


**Bi-classification:**

The models calculate the probability whether two repositories have same topic and the probability is as similarity.

(1) Each single embedding is trained and tested. It verifies that these single embeddings perform well in binary classification tasks.

(2) Repo embedding is used to train the model and there are two models and README file saved in ```TWINS_MODEL```.

The structure is:

![image.png](./assets/1691747644437-image.png)

The effectiveness metric calculates the rate how many same repository pairs can get over 95% probability.

The performance of two models:


| Models | Accuracy | Loss   | Recall           | Precision        | Effectiveness |
| -------- | ---------- | -------- | ------------------ | ------------------ | --------------- |
| 528684 | 83.3%    | 0.3695 | [0.8314, 0.9880] | [0.9139, 0.9938] | 172/200       |
| 157130 | 91.5%    | 0.2128 | [0.9201, 0.9938] | [0.9932, 0.9203] | 71/200        |


## Sequential strategy

In ```sequential``` directory, the code in notebook show all works for sequential strategy. embedding inherently possess sequential information, and utilizing the "mean" strategy may result in a substantial loss of information. In order to harness this information more effectively, a datasets containing sequential information is generated. This is achieved by cutting or padding the length of each embedding to 50, ensuring that the datasets can be utilized for training neural network models. Subsequent models in the following sections utilize this sequential datasets.

The models based on sequential stragety are bi-clssifiers.

A RNN model is designed which consists of GRU and a classifier.


**single embedding**: A Bi-classifier designed to accept unfixed-length embedding accepts the row single embedding and calculate the similarity.

![image.png](./assets/1691748118022-image.png)


**Repo embedding:** A Bi-classifier designed to accept fixed-length embedding accepts the repo embedding whose length is 50.

![image.png](./assets/1691748188851-image.png)


The perdormance of single embedding classifier is not good. 


| Embedding type | Train accuracy | Train loss | Valid accuracy | Valid loss |
| ---------------- | ---------------- | ------------ | ---------------- | ------------ |
| Codes          | 87.5%          | 0.003      | 80.6%          | 0.004      |
| Document       | 48.9%          | 0.005      | 50%            | 0.005      |
| requirement    | 50%            | 0.005      | 50.1%          | 0.005      |
| README         | 48%            | 0.005      | 50.1%          | 0.005      |


However, the performance of repo embedding classifier is close to mean stregety.


| Accuracy | Loss   | Recall           | Precision        | Effectiveness |
| ---------- | -------- | ------------------ | ------------------ | --------------- |
| 88.9%    | 0.2613 | [0.8947, 0.9888] | [0.9877, 0.9038] | 55/200        |


## Calculate Similarity

The similarity here actually represents the probability determined by the binary classifier that two repositories share the same topic. However, due to the nature of the twins neural network model, to ensure consistent results regardless of the input order, the final similarity is the average of the outputs obtained by feeding two repository embedding with different orders. In addition, before proceeding, the names of the two repositories are compared, and if they are the same, the similarity is directly set to 1.

![image.png](./assets/1691748353138-image.png)
