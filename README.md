# BloodPressure-Prediction
## Abstract

Arterial blood pressure (ABP) is a necessary biomedical signal that is used to gauge patient risk for various cardiovascular events. However, due to difficulties obtaining ABP measurements in certain patient demographics, electrocardiogram (ECG) and photoplethysmography (PPG) signals have been collected in lieu of ABP data. Our paper identifies a useful LSTM architecture that can predict ABP measurements from ECG and PPG patient data and demonstrates how generative adversarial networks (GANs) can be utilized to augment ABP training dataset for future models.

## Introduction

Every organ and tissue in the body requires blood in order to function as it is the method of transportation for nutrients, hormones, and other molecular compounds. Blood is pushed through the large vessels of the circulatory system via the heart, exerting pressure on the walls of the vessels as it passes. This pressure, known as arterial blood pressure (ABP), or simply blood pressure (BP), is commonly monitored by physicians and medical staff to gauge a person’s risk for cardiovascular events such as a myocardial infarction, arterial disease, or even stroke [1]. These determinations are possible because arterial blood pressure is made up of two values, a systolic pressure and a diastolic pressure. The former describes the pressure exerted from the contractile force of the heart while the latter represents the stress induced by the relaxation of the cardiac muscle [2]. 

The traditional method for measuring arterial blood pressure is via mercury sphygmomanometer, a process that requires the patient to wrap a cuff around the brachial artery and inflate air to pressurize the cuffed area until blood flow stops. While this procedure is considered the gold standard for determining blood pressure, it is heavily prone to human error and subjectivity, and is particularly inaccurate for obese patients [3], [4]. This method has since been modernized to employ oscillometry, but its accuracy is still poor for patients with large arm circumferences [4]. This can result in physicians employing other modalities in diagnosing cardiovascular diseases, such as electrocardiograms (ECG) and photoplethysmography (PPG).

An electrocardiogram is a reading of the electrical activity of the heart. Given that it’s non-invasive and easy to employ, it has become the most substantial method for investigating the presence of arrhythmias, ischemias, and numerous other cardiovascular diseases [5]. Unlike abnormalities found in ABP readings of obese patients [4], ECGs show much smaller, albeit not negligible, errors and inconsistencies within the collected data [6]. Due to this, clinicians in the hospital setting virtually always collect and monitor ECG information as it is extremely reliable in providing aid to diagnose critical cardiovascular conditions.

Alongside the ECG, most clinics also introduce PPG in their cardiovascular and circulatory patient monitoring routine. PPG uses a light source that is shined through the skin of a patient with a photodetector on the other side; the absorption of the light by the vessel determines the volumetric change of the circulating blood along with the oxygen saturation in the red blood cells (RBCs) [7], [8]. Furthermore, its ability to be easily interpreted and inexpensiveness has made it an excellent tool for patient care. Recent research has also suggested that PPG readings can reveal the presence of cardiovascular diseases, such as arterial stiffness [7], but it is still pending. Although examining the results of a PPG in isolation is not necessarily sufficient to detect cardiovascular disease, pairing the data with ECG recordings proves to be sufficient for diagnosing the presence of risk factors such as hypertension [9].

In this paper, we introduce deep learning model architectures that will be examined in its ability to predict an ABP reading. The work is focused on testing various hyperparameters within a long short-term memory (LSTM)  and dense net architecture, with the goal being to utilize a model that is capable of predicting the ABP of a patient given their PPG and ECG data. With ABP data being difficult to obtain for obese patients and ones with large arm circumferences [4], we hope to alleviate this issue using data that is easily accessible among all patient types. We also propose a generative adversarial network (GAN) that is used to create augmented ABP data that could be used to increase the size and availability of the training dataset.


## Materials & Methods

To identify which architecture would be best suited for ABP prediction, multiple deep learning architectures were explored. For our analysis, we utilized the TensorFlow and the Keras packages due to their ease of use in the Google Colab system.

### LSTM

The first model was a many-to-many LSTM. The architecture of the LSTM can be found in Figure 1. Different hyperparameter values {128, 256, 512} for the number of LSTM units were tested. The ABP signal was normalized to the range [0, 1].

![e](images/LSTM_Architecture.png)

The LSTM model is favorable due to its ability to take in temporal data. PPG, ABP, and ECG signals all have a crucial temporal component.

### LSTM + Dense

An LSTM layer was combined with a dense layer with a linear activation as seen in Figure 2. This modification in the architecture was added to evaluate if the model would predict better ABP signals if it sees the entire input before it starts to output values. Different hyperparameter values of the number of LSTM units were changed to determine the best hyperparameter for the model. The two hyperparameter values that were tested were 256 and 512 LSTM units for the LSTM+Dense model. The ABP signal was normalized to the range [0, 1].


![e](images/LSTM_Dense_Architecture.png)

### Dense

A simple deep neural network was evaluated as well. The deep neural network architectures can be seen in Figure 3a and 3b. Both models use dense layers with higher dimensionalities. This allows the model to act as an autoencoder and denoise the input data. Additionally, the decoder/encoder structure of the model in Figure 3a was designed to act as a transcoder between the different biomedical signals.

![e](images/Dense_Model_1.png)

![e](images/Dense_Model_2.png)

### GAN

The Generative Adversarial Networks (GANs) [10] has proved to be a good resource that can help most models perform better by supplying them with additional information while generating artificial data of high quality. Moreover, the GANs have proven to be useful in analyzing datasets since they are particularly made to recreate trends found in real datasets. Therefore, we would expect the dataset to be hard to analyze if the GAN does not properly and effectively imitate the real data.

The implementation of a GAN in our analysis of Kaggle’s Cuff-Less Blood Pressure Estimation outputs a series of artificially created signals of two kinds: 1) The concatenation of PPG and ECG signals; 2) The predicted ABP signal based on the previous signal.

At the preprocessing level, the data was normalized by creating the following map:

![e](images/GAN_equation.png)
where x and y are min(concat(ppg, ecg) and max(concat(ppg, ecg) respectively. The map places each value from the dataset in the range [-1, 1].

The initial GAN architecture was trained with      10 000 epochs in order to determine the most efficient number of epochs. The generator model was constructed by adding 3 consecutive sets of Dense layer, Leaky ReLu activation, and a Batch Normalization. As we analyze Figure 7 on the loss functions of the training, the GAN performs the best when trained with approximately 2900 epochs. The clustering of the discriminator and generator signals shows us that the training between 2700 and 3100 epochs proves to be the most efficient having significant overlap as well as some of the lowest values for the loss. Furthermore, we can infer that this is actually true by analyzing the fluctuations of the loss values which tend to rise up to 13 times for the discriminator after rising to 6000 epochs. The training time of the model is also important to note, as seen in Figure 8. The training time for the neural network tends to stagnate for the time period when the discriminator’s loss lowers and converges towards the generator’s. This trend is broken after a certain amount of time has passed and, more precisely, around epoch 3100. This speaks that the discriminator starts to overfit making it ineffective. Analyzing the time graph confirms this as considerably more time is needed to train the network after epoch 3130.

The previous observations prompted the retraining of the network using 2900 epochs which led to the graphs shown on Figure 10.

## Results

Using the models highlighted in the Materials and Methods section above, we are able to predict ABP signals from ECG and PPG at varying capacities. An example of the prediction outputs from the LSTM-only architectures can be seen in Figure 4. An example of the prediction outputs from the LSTM+Dense models can be found in Figure 5. An example of the prediction outputs from the Dense-only model can be found in Figure 6. The x-axis for all of these figures represent time. The y-axis for all of the figures represents blood pressure after being normalized to [0, 1].

![e](images/Model_Predictions.png)

As we can see on Figure 10, the results from the GAN trained with 2900 epochs are in much closer proximity than those from the GAN trained with 10000 epochs, ranging from 0.02 to 0.52 as opposed to (-0.8)  (1.0). Additionally, Figure 9 shows us the predictions from the GAN and we can recognize that the general trends are preserved, their only difference with the original artificially created signal being spatially translated to the right.

![e](images/GAN_Model.png)

![e](images/MSE_GAN_Loss.png)

## Conclusion

From the prediction results seen in Figure 4b and the acquired MSE loss values, the model that is best suited for this problem is the many-to-many LSTM model with 512 LSTM units. The LSTM model was able to predict both the amplitude and the periodicity of the ABP signal. The LSTM+Dense demonstrated limited ability to predict the amplitude at later timesteps, but the overall prediction signal is noisy with a lack of clear periodicity. The Dense-only models, in contrast, were only able to capture the periodicity data and could not predict the amplitude of the ABP signal.

The generator model created by the GAN managed to produce models which resemble the initial dataset. The resemblance can be recognized by noticing the peaks in the graphs and their regular occurrence throughout almost even intervals. Compared to the real data, the GAN lacks the smoothness of the curves between each set of local maxima. However, this can be accounted for by acknowledging that the initial size of the training data was not big and, therefore, not allowing the model to refine its features.

Predicting the ABP of obese patients will greatly aid clinical staff in accurately monitoring, and even potentially diagnosing, cardiovascular diseases. While this paper focuses on developing lstm and dense network architectures in predicting ABP values, it fails to produce an adequate curve that mimics the ground truth. Future works can further improve the accuracy and representation of the data by potentially utilizing a gated recurrent unit (GRU) mechanism or an echo state network (ESN) architecture. Furthermore, it may be beneficial to also pursue attempting to predict the systolic and diastolic components of ABP as opposed to all of the intermediary values.	

## References

[1]	W. S. Aronow, “Measurement of blood pressure,” Ann. Transl. Med., vol. 5, no. 3, pp. 49–49, Feb. 2017, doi: 10.21037/atm.2017.01.09.
[2]      	N. C. for B. Information, U. S. N. L. of M. 8600 R. Pike, B. MD, and 20894 Usa, What is blood pressure and how is it measured? Institute for Quality and Efficiency in Health Care (IQWiG), 2019. Accessed: Dec. 14, 2021. [Online]. Available: https://www.ncbi.nlm.nih.gov/books/NBK279251/
[3]      	G. Ogedegbe and T. Pickering, “Principles and techniques of blood pressure measurement,” Cardiol. Clin., vol. 28, no. 4, pp. 571–586, Nov. 2010, doi: 10.1016/j.ccl.2010.07.006.
[4]      	P. Palatini, “Blood pressure measurement in the obese: still a challenging problem,” E-J. Cardiol. Pract., vol. 16, no. 21, Aug. 2018, Accessed: Dec. 14, 2021. [Online]. Available: https://www.escardio.org/Journals/E-Journal-of-Cardiology-Practice/Volume-16/Blood-pressure-measurement-in-the-obese-still-a-challenging-problem
[5]      	Y. Sattar and L. Chhabra, “Electrocardiogram,” in StatPearls, Treasure Island (FL): StatPearls Publishing, 2021. Accessed: Dec. 14, 2021. [Online]. Available: http://www.ncbi.nlm.nih.gov/books/NBK549803/
[6]      	I. Eisenstein, J. Edelstein, R. Sarma, M. Sanmarco, and R. H. Selvester, “The electrocardiogram in obesity.,” J. Electrocardiol., vol. 15, no. 2, pp. 115–118, Apr. 1982, doi: 10.1016/s0022-0736(82)80003-4.
[7]      	D. Castaneda, A. Esparza, M. Ghamari, C. Soltanpur, and H. Nazeran, “A review on wearable photoplethysmography sensors and their potential future applications in health care,” Int. J. Biosens. Bioelectron., vol. 4, no. 4, pp. 195–202, 2018, doi: 10.15406/ijbsbe.2018.04.00125.
[8]      	J. Allen, “Photoplethysmography and its application in clinical physiological measurement.,” Physiol. Meas., vol. 28, no. 3, pp. R1-39, Mar. 2007, doi: 10.1088/0967-3334/28/3/R01.
[9]	Y. Liang, Z. Chen, R. Ward, and M. Elgendi, “Hypertension Assessment via ECG and PPG Signals: An Evaluation Using MIMIC Database,” Diagn. Basel Switz., vol. 8, no. 3, p. 65, Sep. 2018, doi: 10.3390/diagnostics8030065.
[10]	A. Creswell, T. White, V. Dumoulin, K. Arulkumaran, B. Sengupta and A. A. Bharath, "Generative Adversarial Networks: An Overview," in IEEE Signal Processing Magazine, vol. 35, no. 1, pp. 53-65, Jan. 2018, doi: 10.1109/MSP.2017.2765202.

## Data

https://www.kaggle.com/mkachuee/BloodPressureDataset
