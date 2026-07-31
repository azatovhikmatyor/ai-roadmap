**Arithmetic mean**
   $$
   \bar{X} = \frac{\sum_{i=1}^n x_i}{n}
   $$

---

**Geometric Mean**: Used for data involving growth rates:
   $$
   \text{Geometric Mean} = \left( \prod_{i=1}^n x_i \right)^{\frac{1}{n}}
   $$

---

**Harmonic Mean**: Used for rates and ratios:
   $$
   \text{Harmonic Mean} = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}
   $$

---

**Variance**

$$
\text{Var}(X) = \sigma_X^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{X})^2 
$$

---

**Covariance**

$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{X})(y_i - \bar{Y})
$$

---

**Correlation**

$$
\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

Where \( \sigma_X \) and \( \sigma_Y \) are the standard deviations of \( X \) and \( Y \).

---

**Determination (R²):**
   $$
   R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
   $$

Where:

- \( y_i \) is the actual value,
- \( \hat{y}\_i \) is the predicted value,
- \( \bar{y} \) is the mean of the actual values.

---

**Mean Squared Error (MSE):**
   $$
   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   $$

---

**Mean Absolute Error (MAE):**
   $$
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   $$

---

**Slope (\(b_1\)):**
$$
b_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{X})(y_i - \bar{Y})}{\sum_{i=1}^{n} (x_i - \bar{X})^2} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}
$$

---

**Intercept (\(b_0\)):**
$$
b_0 = \bar{Y} - b_1 \bar{X}
$$

---

**L1 norm** of a vector, also known as the **Manhattan norm** or **Taxicab norm**.
For a vector \( \mathbf{v} = [v_1, v_2, \dots, v_n] \), the L1 norm is given by:

$$
\|\mathbf{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n|
$$

---

**L2 norm** of a vector, also called the **Euclidean norm**.
For a vector \( \mathbf{v} = [v_1, v_2, \dots, v_n] \), the L2 norm is calculated as:

$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
$$

---

**Sigmoid Function**  
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

For a vector \( \mathbf{z} = [z_1, z_2, \ldots, z_n] \).

---

**Step Function**  
\[
f(z) =
\begin{cases} 
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
\]

---

**Softmax Function**  
\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
\]

---

**Linear Regression**
$$
\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n
$$


**Matrix Form**  

\[
\hat{y} = X\theta
\]

**\(\theta\)**: Parameter Vector  
\[
\theta =
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2 \\
\vdots \\
\theta_n
\end{bmatrix}
\]  

**\(X\)**: Feature Matrix
\[
X =
\begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
\end{bmatrix}
\]  

---
**Cost function for Linear Regression**
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( \hat{y}_i - y_i \right)^2
$$


**Derivative**
$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
$$

**Matrix Form**
$$
\nabla_\theta J(\theta) = \frac{1}{m} X^\top (X\theta - y)
$$

---
**Gradient Descent and the Cost Function:**

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

---

**Logistic Regression**
\[
\hat{y} = \frac{1}{1 + e^{-\left( \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n \right)}}
\]

**Matrix Form**
\[
\hat{y} = \sigma(X\theta) = \frac{1}{1 + e^{-X\theta}}
\]  

---

**Log Loss for single instance**
$$
c(\theta) = \begin{cases} 
-\log(\hat{p}) & \text{if } y = 1 \\
-\log(1-\hat{p}) & \text{if } y = 0 
\end{cases}
$$


**Log Loss** / **Logarithmic Loss** / **Binary Cross-Entropy Loss**
$$
\mathcal{J}(\theta) = -\frac{1}{m}\sum_{i=1}^m\left[y_i\log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \right]
$$

**Derivative**
$$
\frac{\partial}{\partial \theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m(\sigma(\theta^Tx_i) - y_i)x_i
$$

**Matrix Form**
\[
\nabla_\theta J(\theta) = \frac{1}{m} X^\top \left( \sigma(X\theta) - y \right)
\]

---


Below is a comprehensive list of commonly used **Machine Learning** and **Deep Learning loss functions** along with their formulas. These are grouped into categories based on the type of problem they address:


## **1. Regression Loss Functions**
Used for problems where the target is continuous.

### a. **Mean Squared Error (MSE):**
\[
\text{MSE} = \frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2
\]
- \( y_i \): True value.
- \( \hat{y}_i \): Predicted value.
- \( N \): Number of samples.

### b. **Mean Absolute Error (MAE):**
\[
\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| y_i - \hat{y}_i \right|
\]

### c. **Huber Loss:**
\[
L = \begin{cases} 
\frac{1}{2} \left( y_i - \hat{y}_i \right)^2 & \text{if } \left| y_i - \hat{y}_i \right| \leq \delta \\
\delta \left| y_i - \hat{y}_i \right| - \frac{1}{2} \delta^2 & \text{if } \left| y_i - \hat{y}_i \right| > \delta 
\end{cases}
\]
- Combines MSE and MAE for robustness to outliers.

### d. **Log-Cosh Loss:**
\[
L = \sum_{i=1}^N \log \left( \cosh \left( y_i - \hat{y}_i \right) \right)
\]
- Less sensitive to outliers than MSE.

### e. **Quantile Loss:**
\[
L = \sum_{i=1}^N \max \left( q \cdot (y_i - \hat{y}_i), (q-1) \cdot (y_i - \hat{y}_i) \right)
\]
- \( q \): Quantile to be predicted (e.g., 0.5 for median).

---

## **2. Classification Loss Functions**
Used for problems where the target is categorical.

### a. **Binary Cross-Entropy (Log Loss):**
\[
\text{BCE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]
- For binary classification tasks.

### b. **Categorical Cross-Entropy:**
\[
\text{CCE} = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^C y_{ij} \log(\hat{y}_{ij})
\]
- \( C \): Number of classes.
- Used for multi-class classification.

### c. **Hinge Loss:**
\[
L = \frac{1}{N} \sum_{i=1}^N \max \left( 0, 1 - y_i \hat{y}_i \right)
\]
- Used for SVMs and margin-based classifiers.

### d. **Kullback-Leibler (KL) Divergence:**
\[
\text{KL}(P \| Q) = \sum_{i=1}^N P(x_i) \log \frac{P(x_i)}{Q(x_i)}
\]
- Measures the difference between two probability distributions.

---

## **3. Ranking Loss Functions**
Used for ranking tasks (e.g., information retrieval).

### a. **Hinge Ranking Loss:**
\[
L = \max(0, 1 - s_{y_i} + s_{y_j})
\]
- \( s_{y_i} \): Score for a positive instance.
- \( s_{y_j} \): Score for a negative instance.

### b. **Contrastive Loss:**
\[
L = \frac{1}{2N} \sum_{i=1}^N \left[ y_i \cdot D^2 + (1 - y_i) \cdot \max(0, \text{margin} - D)^2 \right]
\]
- \( D \): Distance between pairs of embeddings.

---

## **4. Probabilistic Loss Functions**
Used for probabilistic models.

### a. **Negative Log Likelihood (NLL):**
\[
L = -\frac{1}{N} \sum_{i=1}^N \log(P(y_i | \hat{y}_i))
\]

### b. **Poisson Loss:**
\[
L = \frac{1}{N} \sum_{i=1}^N \left( \hat{y}_i - y_i \log(\hat{y}_i) \right)
\]

---

## **5. Custom Loss Functions for Neural Networks**
Used for specific tasks or architectures.

### a. **Dice Loss:**
\[
L = 1 - \frac{2 \cdot |X \cap Y|}{|X| + |Y|}
\]
- Commonly used in image segmentation.

### b. **IoU (Intersection over Union) Loss:**
\[
L = 1 - \frac{\text{Intersection}}{\text{Union}}
\]

### c. **Focal Loss:**
\[
L = -\frac{1}{N} \sum_{i=1}^N \alpha (1 - \hat{y}_i)^\gamma \log(\hat{y}_i)
\]
- \( \gamma \): Focusing parameter to down-weight easy examples.

---

## **6. Generative Model Loss Functions**
Used in GANs and VAEs.

### a. **Adversarial Loss (GANs):**
\[
L = \mathbb{E}[\log(D(x))] + \mathbb{E}[\log(1 - D(G(z)))]
\]

### b. **Reconstruction Loss (VAEs):**
\[
L = \|x - \hat{x}\|^2 + \text{KL}(q(z|x) \| p(z))
\]

---

## **7. Reinforcement Learning Loss Functions**
Used for policy optimization.

### a. **Policy Gradient Loss:**
\[
L = -\mathbb{E}[\log \pi(a|s) \cdot R]
\]
- \( R \): Reward.

### b. **Temporal Difference (TD) Loss:**
\[
L = \left( R + \gamma V(s') - V(s) \right)^2
\]

---

## Summary Table:
| **Category**         | **Loss Function**               | **Formula**                           |
|-----------------------|---------------------------------|---------------------------------------|
| Regression            | MSE, MAE, Huber, Log-Cosh      | Handles continuous targets.           |
| Classification        | BCE, CCE, Hinge, KL Divergence | Handles categorical targets.          |
| Ranking               | Hinge Ranking, Contrastive     | For ranking/retrieval tasks.          |
| Probabilistic         | NLL, Poisson                  | Probabilistic models.                 |
| Custom (Neural Nets)  | Dice, IoU, Focal              | Specific neural net tasks.            |
| Generative Models     | Adversarial, Reconstruction    | GANs and VAEs.                        |
| Reinforcement Learning| Policy Gradient, TD           | For RL optimization.                  |


---

### **Activation Functions**  

**ReLU (Rectified Linear Unit)**  
\[
f(z) = \max(0, z)
\]  

**Leaky ReLU**  
\[
f(z) = \begin{cases} 
z & \text{if } z \geq 0 \\ 
\alpha z & \text{if } z < 0 
\end{cases}
\]  
Where \( \alpha \) is a small positive constant.  

**Tanh (Hyperbolic Tangent)**  
\[
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
\]  

---

### **Loss Functions**  

**Mean Squared Logarithmic Error (MSLE)**  
\[
\text{MSLE} = \frac{1}{n} \sum_{i=1}^n \left( \log(1 + y_i) - \log(1 + \hat{y}_i) \right)^2
\]  

**Hinge Loss**  
\[
\mathcal{L}(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})
\]  

**Categorical Cross-Entropy**  
\[
\mathcal{L} = - \sum_{i=1}^k y_i \log(\hat{p}_i)
\]  
Where \( k \) is the number of classes, \( y_i \) is a one-hot encoded vector, and \( \hat{p}_i \) is the predicted probability for class \( i \).  

---

### **Regularization**  

**L1 Regularization (Lasso Regression)**  
\[
\mathcal{J}(\theta) = J(\theta) + \lambda \|\theta\|_1 = J(\theta) + \lambda \sum_{j=1}^n |\theta_j|
\]  

**L2 Regularization (Ridge Regression)**  
\[
\mathcal{J}(\theta) = J(\theta) + \lambda \|\theta\|_2^2 = J(\theta) + \lambda \sum_{j=1}^n \theta_j^2
\]  

**Elastic Net Regularization**  
\[
\mathcal{J}(\theta) = J(\theta) + \lambda_1 \|\theta\|_1 + \lambda_2 \|\theta\|_2^2
\]  

---

### **Optimization Algorithms**  

**Stochastic Gradient Descent (SGD)**  
\[
\theta := \theta - \alpha \nabla_\theta J(\theta)
\]  

**Momentum Update**  
\[
v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta J(\theta)
\]  
\[
\theta := \theta - \alpha v_t
\]  

**Adam Optimizer**  
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta)
\]  
\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta J(\theta))^2
\]  
\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]  
\[
\theta := \theta - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]  

---

### **Neural Networks**  

**Forward Propagation**  
\[
a^{[l]} = \sigma\left(W^{[l]} a^{[l-1]} + b^{[l]}\right)
\]  

**Backward Propagation (Gradient Computation)**  
\[
\frac{\partial J}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^\top
\]  
\[
\frac{\partial J}{\partial b^{[l]}} = \delta^{[l]}
\]  

---

### **Convolutional Neural Networks (CNNs)**  

**Convolution Operation**  
\[
s(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} x(i+m, j+n) \cdot w(m, n) + b
\]  
Where \( x \) is the input, \( w \) is the filter/kernel, \( b \) is the bias term, and \( k \) is the filter size.  

---

### **Evaluation Metrics**  

**Precision**  
\[
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
\]  

**Recall**  
\[
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
\]  

**F1 Score**  
\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]  

**ROC AUC Score**  
\[
\text{AUC} = \int_{0}^{1} TPR \, d(FPR)
\]  


---

### **Support Vector Machines (SVM)**  

**Decision Boundary**  
\[
f(x) = w^\top x + b
\]  

**Hinge Loss for SVM**  
\[
\mathcal{L}(w, b) = \frac{1}{n} \sum_{i=1}^n \max(0, 1 - y_i (w^\top x_i + b)) + \frac{\lambda}{2} \|w\|_2^2
\]  

---

### **Principal Component Analysis (PCA)**  

**Covariance Matrix**  
\[
\Sigma = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^\top
\]  

**Eigenvalue Decomposition**  
\[
\Sigma v = \lambda v
\]  

Where \( \lambda \) represents the eigenvalues and \( v \) represents the eigenvectors.  

---

### **k-Nearest Neighbors (k-NN)**  

**Distance Metric: Euclidean Distance**  
\[
d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
\]  

**Distance Metric: Manhattan Distance**  
\[
d(x, y) = \sum_{i=1}^n |x_i - y_i|
\]  

---

### **Clustering Algorithms**  

#### **K-Means Clustering**  

**Centroid Update**  
\[
\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
\]  
Where \( C_k \) is the cluster of data points assigned to centroid \( k \).  

**Within-Cluster Sum of Squares (WCSS)**  
\[
\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2
\]  

#### **Gaussian Mixture Models (GMM)**  

**Gaussian Probability Density Function**  
\[
p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)
\]  

**Expectation Step (E-step)**  
\[
\gamma_{i,k} = \frac{\pi_k p(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j p(x_i | \mu_j, \Sigma_j)}
\]  

**Maximization Step (M-step)**  
\[
\pi_k = \frac{1}{n} \sum_{i=1}^n \gamma_{i,k}
\]  
\[
\mu_k = \frac{\sum_{i=1}^n \gamma_{i,k} x_i}{\sum_{i=1}^n \gamma_{i,k}}
\]  
\[
\Sigma_k = \frac{\sum_{i=1}^n \gamma_{i,k} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_{i=1}^n \gamma_{i,k}}
\]  

---

### **Reinforcement Learning**  

#### **Q-Learning**  

**Q-value Update Rule**  
\[
Q(s, a) := Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
\]  
Where:  
- \( Q(s, a) \): State-action value.  
- \( \alpha \): Learning rate.  
- \( \gamma \): Discount factor.  

#### **Policy Gradient**  

**Objective Function**  
\[
J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^T R_t \log \pi_\theta(a_t | s_t) \right]
\]  

**Gradient Update**  
\[
\theta := \theta + \alpha \nabla_\theta J(\theta)
\]  

---

### **Deep Learning Regularization**  

#### **Dropout Regularization**  

**Dropout Operation**  
\[
a_i^{(l)} = 
\begin{cases} 
0 & \text{with probability } p \\
\frac{a_i^{(l)}}{1-p} & \text{with probability } 1-p
\end{cases}
\]  

---

### **Time Series Analysis**  

**Autoregressive Model (AR)**  
\[
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p} + \epsilon_t
\]  

**Moving Average Model (MA)**  
\[
X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}
\]  

**ARIMA Model**  
\[
X_t = \phi_1 X_{t-1} + \dots + \phi_p X_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}
\]  

---

### **Bayesian Inference**  

**Bayes' Theorem**  
\[
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\]  

---

### **Information Theory**  

**Entropy**  
\[
H(X) = -\sum_{i=1}^n P(x_i) \log P(x_i)
\]  

**KL Divergence**  
\[
D_{KL}(P || Q) = \sum_{i} P(x_i) \log \frac{P(x_i)}{Q(x_i)}
\]  

**Cross-Entropy Loss**  
\[
H(p, q) = -\sum_{i=1}^n p(x_i) \log q(x_i)
\]  

---

### **Natural Language Processing (NLP)**  

**TF-IDF (Term Frequency-Inverse Document Frequency)**  

- **Term Frequency (TF):**  
\[
\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
\]  

- **Inverse Document Frequency (IDF):**  
\[
\text{IDF}(t) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing term } t}\right)
\]  

- **TF-IDF:**  
\[
\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)
\]  


---

### **1. Joint Probability**
\[
P(A, B) = P(A|B)P(B) = P(B|A)P(A)
\]
Where \(P(A, B)\) is the probability of events \(A\) and \(B\) happening together.

---

### **2. Marginal Probability**
\[
P(A) = \sum_B P(A, B)
\]
Where the probability of \(A\) is computed by summing over all possible values of \(B\).

---

### **3. Conditional Probability**
\[
P(A|B) = \frac{P(A, B)}{P(B)} \quad \text{(if } P(B) \neq 0\text{)}
\]

---

### **4. Bayes' Theorem**
\[
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\]
Where \(P(A)\) is the prior, \(P(B|A)\) is the likelihood, and \(P(B)\) is the evidence.

---

### **5. Chain Rule for Joint Probabilities**
\[
P(x_1, x_2, \dots, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2) \dots P(x_n|x_1, x_2, \dots, x_{n-1})
\]

---

### **6. Gaussian Distribution**
\[
P(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
\]
Where:
- \(x\): random variable
- \(\mu\): mean
- \(\sigma^2\): variance

---

### **7. Multivariate Gaussian Distribution**
\[
P(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)
\]
Where:
- \(x\): \(n\)-dimensional vector
- \(\mu\): mean vector
- \(\Sigma\): covariance matrix

---

### **8. Hidden Markov Models (HMMs)**

#### **Forward Algorithm (Recursive Formula)**
\[
\alpha_t(i) = P(O_1, O_2, \dots, O_t, S_t = i) = \sum_{j=1}^N \alpha_{t-1}(j) a_{ji} b_i(O_t)
\]
Where:
- \(a_{ji}\): transition probability from state \(j\) to \(i\)
- \(b_i(O_t)\): emission probability of observation \(O_t\) in state \(i\)

#### **Backward Algorithm (Recursive Formula)**
\[
\beta_t(i) = P(O_{t+1}, O_{t+2}, \dots, O_T | S_t = i) = \sum_{j=1}^N a_{ij} b_j(O_{t+1}) \beta_{t+1}(j)
\]

#### **Viterbi Algorithm**
\[
\delta_t(i) = \max_{j} \left[\delta_{t-1}(j) a_{ji}\right] b_i(O_t)
\]

---

### **9. Expectation-Maximization (EM Algorithm)**

#### **E-Step (Expectation Step)**
Compute the posterior probabilities:
\[
\gamma_{i,k} = P(Z_k = 1 | x_i; \theta) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
\]

#### **M-Step (Maximization Step)**
Update the parameters:
\[
\pi_k = \frac{1}{n} \sum_{i=1}^n \gamma_{i,k}
\]
\[
\mu_k = \frac{\sum_{i=1}^n \gamma_{i,k} x_i}{\sum_{i=1}^n \gamma_{i,k}}
\]
\[
\Sigma_k = \frac{\sum_{i=1}^n \gamma_{i,k} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_{i=1}^n \gamma_{i,k}}
\]

---

### **10. Bayesian Networks**

#### **Joint Probability in Bayesian Networks**
\[
P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P(X_i | \text{Parents}(X_i))
\]

---

### **11. Naive Bayes Classifier**

#### **Prediction Formula**
\[
P(C|X) \propto P(C) \prod_{i=1}^n P(X_i|C)
\]
Where:
- \(P(C)\): Prior probability of class \(C\)
- \(P(X_i|C)\): Likelihood of feature \(X_i\) given class \(C\)

---

### **12. KL Divergence**
\[
D_{KL}(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
\]

---

### **13. Maximum Likelihood Estimation (MLE)**
\[
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n P(x_i | \theta)
\]
Alternatively (log-likelihood):
\[
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log P(x_i | \theta)
\]

---

### **14. Maximum A Posteriori (MAP)**
\[
\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta | x) = \arg\max_\theta P(x | \theta)P(\theta)
\]

---

### **15. Dirichlet Distribution**
\[
P(\theta | \alpha) = \frac{1}{B(\alpha)} \prod_{i=1}^K \theta_i^{\alpha_i - 1}
\]
Where:
- \(\alpha = (\alpha_1, \alpha_2, \dots, \alpha_K)\): Concentration parameters
- \(B(\alpha)\): Beta function (normalization constant)

---

### **16. Latent Dirichlet Allocation (LDA)**

#### **Topic-Word Distribution**
\[
\phi_k = P(w | z = k)
\]

#### **Document-Topic Distribution**
\[
\theta_d = P(z | d)
\]

#### **Posterior Probability of Topic Assignment**
\[
P(z_{d,n} = k | w_{d,n}, \theta, \phi) \propto \theta_{d,k} \phi_{k, w_{d,n}}
\]

