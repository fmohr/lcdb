# LCDB (Learning Curve DataBase)
LCDB is a set of pre-computed learning curves of common learners on a variety of datasets. You can easily access it via an API we describe here.

## Installation

```shell
pip install lcdb
```

## Example Notebook
We *strongly* encourage to check out the example notebook in [python/example-usage.ipynb](https://github.com/fmohr/lcdb/blob/main/python/example-usage.ipynb). It contains all the code of this tutorial, and also shows how to get an overview of all the contained datasets.

## Getting a Learning Curve
By default, you receive the learning curve for accuracy (see below for other metrics). Learning curves contain observations at powers of $\sqrt{2}$, i.e. 16, 24, 32, 45, 64, 91, 128, ... until 90% of the dataset size (also if this is not a power of $\sqrt{2}$.
Since datasets have different sizes, the schedules have different lengths for different datasets, which is why there is no tensor view on the data.

### By Dataset Name
```python
curve = lcdb.get_curve("kr-vs-kp", "sklearn.linear_model.LogisticRegression")
anchors, scores_train, scores_valid, scores_test = curve
```
The first argument is the dataset name or the dataset id of openml.org. The second argument is the name of the learner.

### By OpenML ID
```python
curve = lcdb.get_curve(3, "sklearn.linear_model.LogisticRegression")
anchors, scores_train, scores_valid, scores_test = curve
```

### For Other Metrics
The metric is the third positional argument. Here, we get the curve for log-loss
```python
curve_log_loss = lcdb.get_curve("kr-vs-kp", "sklearn.linear_model.LogisticRegression", "logloss")
```
Currently supported metrics: `accuracy`, `logloss`

We maintain the full prediction vectors and probability distributions offline, so if you want to add some metric, please let us know and we will compute it and add it to the repository.

## Plotting Curves
Of course, you can plot the curves with any tool you want. There are some built-ins though.

### Quicky
```python
lcdb.plot_train_and_test_curve(curve)
```
will give you

![Directly plotted learning curve](/doc/plots/learningcurve-1.png)

### With your own axis objects
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(20, 4))
lcdb.plot_train_and_test_curve(curve, ax[0])
ax[0].set_ylabel("Accuracy")
lcdb.plot_train_and_test_curve(curve_log_loss, ax[1])
ax[1].set_ylabel("Log-Loss")
plt.show()
```
will give you

![Learning Curves plotted into your axies objects](/doc/plots/learningcurve-2.png)

## Training Times
You can also get the training times of the learners.

### Times per Anchor
```python
lcdb.get_train_times("kr-vs-kp", "sklearn.linear_model.LogisticRegression")
```
### Plot Training Times
#### Quicky
```python
lcdb.plot_train_times("kr-vs-kp", "sklearn.linear_model.LogisticRegression")
```
will give you

![Directly plotted runtime](/doc/plots/times-1.png)


#### With your axis objects
```python
fig, ax = plt.subplots()
lcdb.plot_train_times("kr-vs-kp", "sklearn.linear_model.LogisticRegression", ax)
lcdb.plot_train_times("kr-vs-kp", "SVC_linear", ax)
```
will give you

![Customized runtime plots](/doc/plots/times-2.png)

## Meta-Features
All datasets come with pre-computed meta-features. You can retrieve these as follows:

### Dataframe with Meta-Features for all Datasets
```python
lcdb.get_meta_features()
```

### Dictionary with Meta-Features for Specific Dataset
```python
lcdb.get_meta_features("kr-vs-kp") # by name
lcdb.get_meta_features("3) # by openml.org id
```

# Citing LCDB
If you use our database and find it helfpul, please cite the ECML paper:
```
@inproceedings{lcdb,
  title={LCDB 1.0: An Extensive Learning Curves Database for Classification Tasks},
  author={Mohr, Felix and Viering, Tom J and Loog, Marco and van Rijn, Jan N},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track - European Conference, {ECML} {PKDD} 2022, Grenoble, France, September 19-24, 2022},
  year={2022}
}
```

# Supported Learners
These are the 20 supported learners (all of them executed with standard parametrization):
* SVC_linear
* SVC_poly
* SVC_rbf
* SVC_sigmoid
* sklearn.tree.DecisionTreeClassifier
* sklearn.tree.ExtraTreeClassifier
* sklearn.linear_model.LogisticRegression
* sklearn.linear_model.PassiveAggressiveClassifier
* sklearn.linear_model.Perceptron
* sklearn.linear_model.RidgeClassifier
* sklearn.linear_model.SGDClassifier
* sklearn.neural_network.MLPClassifier
* sklearn.discriminant_analysis.LinearDiscriminantAnalysis
* sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
* sklearn.naive_bayes.BernoulliNB
* sklearn.naive_bayes.MultinomialNB
* sklearn.neighbors.KNeighborsClassifier
* sklearn.ensemble.ExtraTreesClassifier
* sklearn.ensemble.RandomForestClassifier
* sklearn.ensemble.GradientBoostingClassifier

# Currently Maintained Datasets
* 3 (kr-vs-kp)
* 6 (letter)
* 12 (mfeat-factors)
* 14 (mfeat-fourier)
* 16 (mfeat-karhunen)
* 18 (mfeat-morphological)
* 21 (car)
* 22 (mfeat-zernike)
* 23 (cmc)
* 24 (mushroom)
* 26 (nursery)
* 28 (optdigits)
* 30 (page-blocks)
* 31 (credit-g)
* 
(pendigits)
* 36 (segment)
* 38 (sick)
* 44 (spambase)
* 46 (splice)
* 54 (vehicle)
* 57 (hypothyroid)
* 60 (waveform-5000)
* 179 (adult)
* 180 (covertype)
* 181 (yeast)
* 182 (satimage)
* 183 (abalone)
* 184 (kropt)
* 185 (baseball)
* 188 (eucalyptus)
* 273 (IMDB.drama)
* 293 (covertype)
* 300 (isolet)
* 351 (codrna)
* 354 (poker)
* 357 (vehicle_sensIT)
* 389 (fbis.wc)
* 390 (new3s.wc)
* 391 (re0.wc)
* 392 (oh0.wc)
* 393 (la2s.wc)
* 395 (re1.wc)
* 396 (la1s.wc)
* 398 (wap.wc)
* 399 (ohscal.wc)
* 401 (oh10.wc)
* 485 (analcatdata_vehicle)
* 554 (mnist_784)
* 679 (rmftsa_sleepdata)
* 715 (fri_c3_1000_25)
* 718 (fri_c4_1000_100)
* 720 (abalone)
* 722 (pol)
* 723 (fri_c4_1000_25)
* 727 (2dplanes)
* 728 (analcatdata_supreme)
* 734 (ailerons)
* 735 (cpu_small)
* 737 (space_ga)
* 740 (fri_c3_1000_10)
* 741 (rmftsa_sleepdata)
* 743 (fri_c1_1000_5)
* 751 (fri_c4_1000_10)
* 752 (puma32H)
* 761 (cpu_act)
* 772 (quake)
* 797 (fri_c4_1000_50)
* 799 (fri_c0_1000_5)
* 803 (delta_ailerons)
* 806 (fri_c3_1000_50)
* 807 (kin8nm)
* 813 (fri_c3_1000_5)
* 816 (puma8NH)
* 819 (delta_elevators)
* 821 (house_16H)
* 822 (cal_housing)
* 823 (houses)
* 833 (bank32nh)
* 837 (fri_c1_1000_50)
* 843 (house_8L)
* 845 (fri_c0_1000_10)
* 846 (elevators)
* 847 (wind)
* 849 (fri_c0_1000_25)
* 866 (fri_c2_1000_50)
* 871 (pollen)
* 881 (mv)
* 897 (colleges_aaup)
* 901 (fried)
* 903 (fri_c2_1000_25)
* 904 (fri_c0_1000_50)
* 910 (fri_c1_1000_10)
* 912 (fri_c2_1000_5)
* 913 (fri_c2_1000_10)
* 914 (balloon)
* 917 (fri_c1_1000_25)
* 923 (visualizing_soil)
* 930 (colleges_usnews)
* 934 (socmob)
* 953 (splice)
* 958 (segment)
* 959 (nursery)
* 962 (mfeat-morphological)
* 966 (analcatdata_halloffame)
* 971 (mfeat-fourier)
* 976 (JapaneseVowels)
* 977 (letter)
* 978 (mfeat-factors)
* 979 (waveform-5000)
* 980 (optdigits)
* 991 (car)
* 993 (kdd_ipums_la_97-small)
* 995 (mfeat-zernike)
* 1000 (hypothyroid)
* 1002 (ipums_la_98-small)
* 1018 (ipums_la_99-small)
* 1019 (pendigits)
* 1020 (mfeat-karhunen)
* 1021 (page-blocks)
* 1036 (sylva_agnostic)
* 1037 (ada_prior)
* 1039 (hiva_agnostic)
* 1040 (sylva_prior)
* 1041 (gina_prior2)
* 1042 (gina_prior)
* 1049 (pc4)
* 1050 (pc3)
* 1053 (jm1)
* 1059 (ar1)
* 1067 (kc1)
* 1068 (pc1)
* 1069 (pc2)
* 1111 (KDDCup09_appetency)
* 1116 (musk)
* 1119 (adult-census)
* 1120 (MagicTelescope)
* 1128 (OVA_Breast)
* 1130 (OVA_Lung)
* 1134 (OVA_Kidney)
* 1138 (OVA_Uterus)
* 1139 (OVA_Omentum)
* 1142 (OVA_Endometrium)
* 1146 (OVA_Prostate)
* 1161 (OVA_Colon)
* 1166 (OVA_Ovary)
* 1216 (Click_prediction_small)
* 1242 (vehicleNorm)
* 1457 (amazon-commerce-reviews)
* 1461 (bank-marketing)
* 1464 (blood-transfusion-service-center)
* 1468 (cnae-9)
* 1475 (first-order-theorem-proving)
* 1485 (madelon)
* 1486 (nomao)
* 1487 (ozone-level-8hr)
* 1489 (phoneme)
* 1494 (qsar-biodeg)
* 1501 (semeion)
* 1515 (micro-mass)
* 1569 (poker-hand)
* 1590 (adult)
* 4134 (Bioresponse)
* 4135 (Amazon_employee_access)
* 4136 (Dexter)
* 4137 (Dorothea)
* 4534 (PhishingWebsites)
* 4538 (GesturePhaseSegmentationProcessed)
* 4541 (Diabetes130US)
* 4552 (BachChoralHarmony)
* 23380 (cjs)
* 23512 (higgs)
* 23517 (numerai28.6)
* 40497 (thyroid-ann)
* 40498 (wine-quality-white)
* 40668 (connect-4)
* 40670 (dna)
* 40685 (shuttle)
* 40691 (wine-quality-red)
* 40701 (churn)
* 40900 (Satellite)
* 40926 (CIFAR_10_small)
* 40971 (collins)
* 40975 (car)
* 40978 (Internet-Advertisements)
* 40981 (Australian)
* 40982 (steel-plates-fault)
* 40983 (wilt)
* 40984 (segment)
* 40996 (Fashion-MNIST)
* 41026 (gisette)
* 41027 (jungle_chess_2pcs_raw_endgame_complete)
* 41064 (convex)
* 41065 (mnist_rotation)
* 41066 (secom)
* 41138 (APSFailure)
* 41142 (christine)
* 41143 (jasmine)
* 41144 (madeline)
* 41145 (philippine)
* 41146 (sylvine)
* 41147 (albert)
* 41150 (MiniBooNE)
* 41156 (ada)
* 41157 (arcene)
* 41158 (gina)
* 41159 (guillermo)
* 41161 (riccardo)
* 41162 (kick)
* 41163 (dilbert)
* 41164 (fabert)
* 41165 (robert)
* 41166 (volkert)
* 41167 (dionis)
* 41168 (jannis)
* 41169 (helena)
* 41946 (Sick_numeric)
* 42732 (sf-police-incidents)
* 42733 (Click_prediction_small)
* 42734 (okcupid-stem)
