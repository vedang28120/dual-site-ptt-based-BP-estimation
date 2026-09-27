# Dual-Camera AI for Cuffless Blood Pressure Screening
## Final Project Abstract

**Problem:** Conventional blood-pressure measurement depends on inflatable cuffs, making frequent and continuous monitoring inconvenient. Existing camera-based approaches using remote photoplethysmography (rPPG) offer a promising alternative, but models trained on a single facial optical signal often struggle to generalize across unseen individuals because vascular properties differ substantially between subjects.

**Innovation:** We propose a smartphone-based cuffless BP screening framework that combines two optical pulse measurements with inertial sensing rather than relying on a single camera waveform. The front camera captures facial rPPG, while the rear camera and torch simultaneously capture a higher-SNR pulse signal from the fingertip. The temporal difference between these signals provides a differential pulse-arrival feature that is less dependent on cardiac timing. The smartphone's accelerometer is additionally used during a controlled change in hand elevation to measure the vascular response to a known hydrostatic pressure perturbation.

**AI Approach:** A lightweight multimodal temporal neural network combines differential pulse timing, waveform morphology, heart-rate features, and inertial measurements to estimate systolic and diastolic BP or BP ranges. The proposed approach is designed for on-device inference, minimizing dependence on external hardware or cloud processing.

**Validation:** The prototype will first be evaluated against reference BP measurements and compared with our baseline facial-rPPG model. Subject-independent evaluation will be used to determine whether the additional camera and physiological perturbation improve generalization to previously unseen users. Signal-quality checks will identify conditions such as motion, poor fingertip contact, low perfusion, and ambient-light variation where the system becomes unreliable.

**Impact:** This project aims to investigate whether commodity smartphones can provide a more accessible form of cuffless BP screening by combining AI with additional physiological information already available from smartphone sensors. The resulting platform could provide a foundation for future longitudinal cardiovascular monitoring and wearable implementations.
