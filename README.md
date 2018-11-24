# Classification with Support Vector Machine

##### Classification of NBA players into 5 positions on the basketball court: SG (shooting guard), PG (point guard), SF (small forward), PF (power forward), and C (center), based on the player's per-game average performance in the 2015-2016 season.
===============================================================================
### Usage:
- ###### >> python NBA_Classifier.py
-  ###### Dataset: NBAStats.csv
### Data Preprocessing:

  - ##### The dataset is imported into the workspace as a pandas-dataframe.
  #####
  - ##### Parameters like ‘Age’ of the player and the ‘Team’, do not contribute much to the classification process. So, these parameters are ignored.

- ##### Feature Selection: Only 80% of features were selected with Scikit-learn’s automatic feature-selection module.
######
### Classifier Parameters:
- ##### Classifier used : Support Vector Machine ( Max Accuracy is achieved by setting gamma parameter to a very low value and C parameter to a higher threshold value for which we get highest accuracy.)
- - ##### Gamma Parameter: gamma=0. 00007 , anything below or above this value worsens the accuracy of the model.
- - ##### C Parameter: C = 1500 , if we increase the C parameter value above 1500 the accuracy almost stays the same. Decreasing C below 1500 worsens the accuracy.
- -  ##### Random State = random_state = 1
- -  ##### Kernel: RBF kernel
- -  ##### Cross-Validation: 10-Fold Stratified Cross Validation scheme
- -   ##### Accuracy achieved with this model is 72.268%
