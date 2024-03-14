# Gausian-Mixture-Model-for-Audio-Classification
## Overview
This repository contains code for implementing a Gaussian Mixture Model (GMM) for audio classification using wavelet transform features. The GMM is trained to classify audio signals into predefined classes, making it useful for tasks such as speech recognition, environmental sound classification.




### Introduction
The Gaussian Mixture Model (GMM) is a probabilistic model commonly used for clustering and classification tasks. In the context of audio classification, GMMs are particularly useful for modeling the distribution of feature vectors extracted from audio signals. The algorithm used here is Expectation-Maximization Algorithm.

### Expectation-Maximization(EM) Algorithm:
The Expectation-Maximization (EM) algorithm is an iterative optimization method that combines different unsupervised machine learning algorithms to find maximum likelihood or maximum posterior estimates of parameters in statistical models that involve unobserved latent variables. The EM algorithm is commonly used for latent variable models and can handle missing data.




### How Expectation-Maximization (EM)  Algorithm Works:
![image](https://github.com/shekharma/Gausiian-Mixture-Model-for-Audio-Classification/assets/122733304/8db68c82-8dcd-4b27-b3f1-bd809dfb1844) | ![image](https://github.com/shekharma/Gausiian-Mixture-Model-for-Audio-Classification/assets/122733304/3023d85b-d0a5-41dd-9595-61ea6e30833c) 
1.  Initialization:
Initially, a set of initial values of the parameters are considered. A set of incomplete observed data is given to the system with the assumption that the observed data comes from a specific model.
2.  E-Step (Expectation Step): In this step, we use the observed data in order to estimate or guess the values of the missing or incomplete data. It is basically used to update the variables. 
     -  Compute the posterior probability or responsibility of each latent variable given the observed data and current parameter estimates.
     -  Estimate the missing or incomplete data values using the current parameter estimates.
     -  Compute the log-likelihood of the observed data based on the current parameter estimates and estimated missing data.
3. M-step (Maximization Step): In this step, we use the complete data generated in the preceding “Expectation” – step in order to update the values of the parameters. It is basically used to update the hypothesis.
    -  Update the parameters of the model by maximizing the expected complete data log-likelihood obtained from the E-step.
    -  This typically involves solving optimization problems to find the parameter values that maximize the log-likelihood.
    -  The specific optimization technique used depends on the nature of the problem and the model being used.
4.  Convergence: In this step, it is checked whether the values are converging or not, if yes, then stop otherwise repeat step-2 and step-3 i.e. “Expectation” – step and “Maximization” – step until the convergence occurs.
  -  Check for convergence by comparing the change in log-likelihood or the parameter values between iterations.
  -  If the change is below a predefined threshold, stop and consider the algorithm converged.
  -  Otherwise, go back to the E-step and repeat the process until convergence is achieved.

### Model Description
A GMM represents the probability distribution of the observed data as a weighted sum of multiple Gaussian distributions. Each Gaussian component represents a cluster in the feature space. The model assumes that the observed data is generated from a mixture of these Gaussian distributions.

Mathematically, the probability density function (PDF) of a GMM is given by:

\[ p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \]

Where:
- \( \mathbf{x} \) is the observed feature vector.
- \( K \) is the number of Gaussian components in the mixture.
- \( \pi_k \) is the weight associated with the \( k \)-th Gaussian component, representing its relative contribution to the mixture.
- \( \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \) is the multivariate Gaussian distribution with mean \( \boldsymbol{\mu}_k \) and covariance matrix \( \boldsymbol{\Sigma}_k \).

### Training
The parameters of a GMM, including the means, covariances, and mixture weights, are typically estimated from the training data using the Expectation-Maximization (EM) algorithm. The EM algorithm iteratively updates the parameters to maximize the likelihood of the observed data under the model.

During training, the GMM learns to represent the underlying distribution of feature vectors in the training data. Each Gaussian component captures a cluster of similar feature vectors, allowing the model to effectively capture the variability present in the data.

### Classification
Once trained, a GMM can be used for classification by computing the likelihood of a given feature vector under each Gaussian component. Classification is typically performed by assigning the feature vector to the component with the highest likelihood or by using techniques such as maximum a posteriori (MAP) estimation.

In the context of audio classification, GMMs are applied to classify audio signals based on features extracted from the signals, wavelet transform coefficients.


## Wavelet Transform in Audio Processing

### Introduction
Wavelet transform is a mathematical technique used to analyze signals in both time and frequency domains simultaneously. In audio processing, wavelet transform provides a way to extract relevant features from audio signals, enabling tasks such as denoising, compression, and classification.

### Time-Frequency Localization
One of the key advantages of wavelet transform in audio processing is its ability to provide localized time-frequency information. Unlike traditional Fourier transform, which provides a global frequency representation of the entire signal, wavelet transform captures both high and low-frequency components at different time points. This is particularly useful for analyzing audio signals with transient or non-stationary characteristics, such as musical instruments, speech, or environmental sounds.

### Feature Extraction
Wavelet transform is commonly used for feature extraction in audio processing tasks. By decomposing the audio signal into different frequency bands at multiple scales, wavelet transform extracts features that capture both temporal and spectral characteristics of the audio signal. These features can include energy distribution across frequency bands, temporal dynamics, and transient events present in the signal.

### Applications
1. **Audio Classification**: Wavelet transform is used to extract discriminative features from audio signals for tasks such as speech recognition, music genre classification, and environmental sound classification. The localized time-frequency information provided by wavelet transform helps in capturing key characteristics of different audio classes.
  
2. **Audio Compression**: Wavelet transform is employed in audio compression algorithms such as MPEG Audio Coding (MP3) to reduce the size of audio files while preserving perceptual quality. By representing the audio signal with a sparse set of wavelet coefficients, redundant information can be efficiently discarded without significant loss of audio quality.

3. **Denoising**: Wavelet transform is utilized for denoising noisy audio signals by separating the signal from unwanted noise components. The ability to localize signal and noise components in both time and frequency domains allows for effective noise reduction while preserving the integrity of the original audio signal.

### Conclusion

In summary, the combination of Wavelet Transform and Gaussian Mixture Model (GMM) presents a robust framework for audio classification tasks. Wavelet Transform enables the extraction of localized time-frequency features from audio signals, addressing the challenges posed by diverse signal characteristics. These features are then effectively modeled using GMMs, capturing the underlying distribution of the data and facilitating accurate classification. Together, Wavelet Transform and GMM offer a powerful approach for analyzing, processing, and classifying audio signals in various real-world applications.

### Reference
1. https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf
2. https://towardsdatascience.com/what-is-wavelet-and-how-we-use-it-for-data-science-d19427699cef
3. https://adityadutt.medium.com/audio-classification-using-wavelet-transform-and-deep-learning-f9f0978fa246
