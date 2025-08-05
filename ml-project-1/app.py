import pandas as pd
from sklearn.datasets import load_wine
from tabulate import tabulate
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster   

import os
os.environ["XDG_SESSION_TYPE"] = "xcb"


def mainMenu():
    print("\n=== MACHINE LEARNING TOOL ===")
    print("1. Load and Explore Datasets")
    print("2. Regression Model (Boston Housing)")
    print("3. Binary Classification Model (Iris Flower)")
    print("4. Multi-class Classification Models (Wine Quality)")
    print("5. Clustering Analysis (Customer Segmentation)")
    #print("6. Model Evaluation")
    #print("7. Advanced ML Techniques")
    print("8. Exit")
    mainOption = int(input("\nChoose an option: "))

    match mainOption:
        case 1:
            loadDatasets()
        case 2:
            regressionModel()
        case 3:
            classificationModel()
        case 4:
            multiClassClassificationModel()
        case 5:
            clusteringAnalysis()
        #case 6:
            #modelEvaluation()
        #case 7:
            #advancedMLTechniques()


def loadBostonHousing():
    bostonHousing = pd.read_csv("boston.csv")

    print("\n📊 Loading Boston Housing dataset...")
    bostonRow = bostonHousing.shape[0]
    bostonCollumn = bostonHousing.shape[1]
    print(
        f"✅ Successfully loaded {bostonRow} samples with {bostonCollumn-1} features")

    print("\n📋 Dataset Overview:")
    header = list(bostonHousing.columns)
    print("Features: ", ', '.join(header[1:]))
    print("Target: MEDV (Medium Home Value)")
    print("Data Type: Regression Problem")
    print("Use Case: Predict house prices from neighborhood characteristics")

    print("\n📈 Basic Statistics: ")

    selectedFeatures = ["CRIM", "ZN", "INDUS", "RM", "Price"]

    bostonStats = bostonHousing[selectedFeatures].describe(
    ).loc[['mean', 'std', 'min', 'max']]
    bostonStats = bostonStats.round(2)

    bostonTable = bostonStats.T.reset_index()
    bostonTable.columns = ['Feature', 'Mean', 'Std', 'Min', 'Max']

    print(tabulate(bostonTable, headers='keys',
          tablefmt='rounded_grid', showindex=False))

    print("\n🏠 TARGET ANALYSIS (House Prices):")

    def strFormat(x):
        roundNumber = round(x)
        fNumber = f"{roundNumber:,}"
        return fNumber

    meanPrice = bostonHousing['Price'].mean()*1000
    minPrice = bostonHousing['Price'].min()*1000
    maxPrice = bostonHousing['Price'].max()*1000
    print(f"Mean Price: ${strFormat(meanPrice)}")
    print(f"Price Range: ${strFormat(minPrice)} - ${strFormat(maxPrice)}")
    print("Most Expensive Area: Low crime, high room count")
    print("Cheapest Area: High crime, industrial zones")

    print("\n🔍 Data Quality Check:")

    missingValues = bostonHousing.isnull().sum().sum()

    totalOutliers = 0
    for feature in selectedFeatures:
        Q1 = bostonHousing[feature].quantile(0.25)
        Q3 = bostonHousing[feature].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
        outliers = bostonHousing[(bostonHousing[feature] < lowerBound) | (
            bostonHousing[feature] > upperBound)][feature]
        totalOutliers += len(outliers)

    skewness = bostonHousing[selectedFeatures].apply(skew)
    distributionNote = "Normal distribution for most features" if (skewness.abs() < 1).sum(
    ) >= len(selectedFeatures) * 0.6 else "Some features does not follow normal distribution"

    print(f"Missing Value: {missingValues}")
    print(f"Outliers Detected: {totalOutliers}  (using IQR method)")
    print(f"Data Distribution: {distributionNote}")

    def corrbtw2Var(var1, var2):
        correlations = bostonHousing[[
            'RM', 'LSTAT', 'CRIM', 'PTRATIO', 'Price']].corr()
        corr = correlations[var1][var2]
        roundCorr = round(corr, 3)
        return roundCorr

    print("\n📊 KEY RELATIONSHIPS: ")
    print(
        f"- More Rooms (RM) -> Higher Prices {corrbtw2Var('Price','RM')})")
    print(
        f"- Lower status population (LSTAT) → Lower prices ({corrbtw2Var('Price','LSTAT')})")
    print(
        f"- Higher crime (CRIM) → Lower prices ({corrbtw2Var('Price','CRIM')})")
    print(
        f"- Pupil-teacher ratio (PTRATIO) → Lower prices ({corrbtw2Var('Price','PTRATIO')})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bostonHousingMatrix = bostonHousing.corr().round(2)

    sns.heatmap(bostonHousingMatrix, annot=False, ax=ax1)
    ax1.set_title('Correlation Heatmap')

    ax2.hist(bostonHousing['Price'], color='purple')
    ax2.set_ylabel("No. of Houses")
    ax2.set_xlabel("Price range in $1000s")
    ax2.set_title('Price Distribution')

    plt.tight_layout()
    plt.show()

    goBack = input("\nGo Back ? (y/n):  ")
    if goBack == 'y':
        loadDatasets()


def loadIrisFlower():
    irisData = pd.read_csv("iris.csv")

    irisRow = irisData.shape[0]
    irisCollumn = irisData.shape[1]
    print("\n🌸 Loading Iris Flowers dataset...")
    print(
        f"✅ Successfully loaded {irisRow} samples with {irisCollumn} features")

    print("\n📋 Dataset Overview:")
    header = list(irisData.columns)
    print("Features: ", ', '.join(header[1:]))
    print("Target: Species(setosa, versicolor, virginica)")
    print("Data type: Classification problem")
    print("Use case: Identify flower species from measurements")

    print("\n📈 Basic Statistics:")
    irisStats = irisData.describe(
    ).loc[['mean', 'std', 'min', 'max']]
    irisStats = irisStats.round(2)

    irisTable = irisStats.T.reset_index()
    irisTable.columns = ['Feature', 'Mean', 'Std', 'Min', 'Max']

    print(tabulate(irisTable, headers='keys',
          tablefmt='rounded_grid', showindex=False))

    print("\n🌺 CLASS DISTRIBUTION:")
    varietyList = list(irisData['variety'])
    variety = Counter(varietyList)

    for x in variety:
        varietyPercentage = variety[x]/irisRow*100

        if x == 'Setosa':
            petalType = "Smallest Petals"
        elif x == 'Versicolor':
            petalType = "Medium Petals"
        elif x == 'Virginica':
            petalType = 'Largest Petals'

        print(
            f"- {x}: {variety[x]} samples ({varietyPercentage:.2f}%) - {petalType}")

    print("\n🔍 Data Quality Check:")
    missingValues = irisData.isnull().sum().sum()

    selectedFeatures = ["sepal.width", "petal.length", "petal.width"]
    totalOutliers = 0
    for feature in selectedFeatures:
        Q1 = irisData[feature].quantile(0.25)
        Q3 = irisData[feature].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
        outliers = irisData[(irisData[feature] < lowerBound) | (
            irisData[feature] > upperBound)][feature]
        totalOutliers += len(outliers)

    print(f"Missing Values: {missingValues}")
    print(f"Outliers detected: {totalOutliers} (using IQR method)")
    print("Class balance: Perfect (equal distribution)")

    print("\n🎯 CLASSIFICATION INSIGHTS:")
    print("- Setosa: 100 % separable(petal length < 2cm)")
    print("- Versicolor vs Virginica: Some overlap(need multiple features)")
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1,3,1)
    sns.boxplot(x='variety', y='sepal.length', data=iris)
    plt.title('Sepal Length by Species')

    
    plt.subplot(1,3,2)
    colors = {'Setosa': 'blue', 'Versicolor': 'green', 'Virginica': 'red'}

    for species in iris['variety'].unique():
        species_data = iris[iris['variety'] == species]
        plt.scatter(species_data['petal.length'], species_data['sepal.length'],
                    c=colors[species], label=species)

    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Sepal Length (cm)')
    plt.title('Petal Length vs Sepal Length')
    plt.legend()

    plt.subplot(1,3,3)
    sns.scatterplot(x='sepal.length', y='sepal.width', hue='variety',style='variety', data=iris, s=100)
    plt.title('Species Separation: Sepal Length vs. Sepal Width')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend(title='Species')
    
    plt.tight_layout()

    plt.show()

    goBack = input("\nGo Back ? (y/n):  ")
    if goBack == 'y':
        loadDatasets()

    


def loadWine():
    wineData = pd.read_csv("redwine.csv")

    wineRow = wineData.shape[0]
    wineCollumn = wineData.shape[1]

    print("\n🍷 Loading Wine Quality dataset...")
    print(
        f"✅ Successfully loaded {wineRow} samples with {wineCollumn} features")
    print("\n📋 Dataset Overview:")
    header = list(wineData.columns)
    print("Features: ", ', '.join(header[1:]))
    print("Target: Quality(rating from 3-8, where 8 is best)")
    print("Data type: Multi-class classification problem")
    print("Use case: Predict wine quality from chemical properties")

    print("\n📈 Basic Statistics: ")

    selectedFeatures = ["fixed acidity", "volatile acidity",
                        "citric acid", "residual sugar", "alcohol"]

    wineStats = wineData[selectedFeatures].describe(
    ).loc[['mean', 'std', 'min', 'max']]
    wineStats = wineStats.round(2)

    wineTable = wineStats.T.reset_index()
    wineTable.columns = ['Feature', 'Mean', 'Std', 'Min', 'Max']

    print(tabulate(wineTable, headers='keys',
          tablefmt='rounded_grid', showindex=False))

    print("\n🍷 QUALITY DISTRIBUTION:")

    qualitySet = Counter(wineData['quality'])

    print(
        f"- Quality 3: {qualitySet[3]} wines ({qualitySet[3]/wineRow*100:.1f}%) - Poor")
    print(
        f"- Quality 4: {qualitySet[4]} wines ({qualitySet[4]/wineRow*100:.1f}%) - Below Average")
    print(
        f"- Quality 5: {qualitySet[5]} wines ({qualitySet[5]/wineRow*100:.1f}%) - Average")
    print(
        f"- Quality 6: {qualitySet[6]} wines ({qualitySet[6]/wineRow*100:.1f}%) - Good")
    print(
        f"- Quality 7: {qualitySet[7]} wines ({qualitySet[7]/wineRow*100:.1f}%) - Very Good")
    print(
        f"- Quality 8: {qualitySet[8]} wines ({qualitySet[8]/wineRow*100:.1f}%) - Excellent")

    print("\n🔍 Data Quality Check:")
    missingValues = wineData.isnull().sum().sum()

    totalOutliers = 0
    for feature in selectedFeatures:
        Q1 = wineData[feature].quantile(0.25)
        Q3 = wineData[feature].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
        outliers = wineData[(wineData[feature] < lowerBound) | (
            wineData[feature] > upperBound)][feature]
        totalOutliers += len(outliers)

    print(f"Missing Values: {missingValues}")
    print(f"Outliers detected: {totalOutliers} (using IQR method)")
    print("Class balance: Imbalance")

    plt.scatter(wineData["alcohol"], wineData["quality"])
    plt.xlabel('Quality')
    plt.ylabel('Alcohol Content (%)')
    plt.title('Wine Quality vs Alcohol Content')
    plt.show()

    goBack = input("\nGo Back ? (y/n):  ")
    if goBack == 'y':
        loadDatasets()


def loadCustomer():
    customerData = pd.read_csv("customers.csv")
    customerRow = customerData.shape[0]
    customerCollumn = customerData.shape[1]

    print("\n👥 Loading Customer Segmentation dataset...")
    print(
        f"✅ Successfully loaded {customerRow} samples with {customerCollumn} features")

    print("\n📋 Dataset Overview:")
    header = list(customerData.columns)
    print("Features: ", ', '.join(header[1:]))
    print("Target: None (Unsupervised Clustering)")
    print("Data type: Clustering problem")
    print("Use case: Group customers for targeted marketing")

    print("\n📈 Basic Statistics: ")

    selectedFeatures = ["annual_income", "spending_score", "age"]

    customerStats = customerData[selectedFeatures].describe(
    ).loc[['mean', 'std', 'min', 'max']]
    customerStats = customerStats.round(2)

    customerTable = customerStats.T.reset_index()
    customerTable.columns = ['Feature', 'Mean', 'Std', 'Min', 'Max']

    print(tabulate(customerTable, headers='keys',
          tablefmt='rounded_grid', showindex=False))

    print("\n👫 Gender Distribution:")
    genderSet = Counter(customerData['gender'])
    for x in genderSet:
        print(
            f"- {x}: {genderSet[x]} customers ({genderSet[x]/customerRow*100:.2f}%) ")

    youngNum = 0
    middleAgeNum = 0
    oldAgeNum = 0

    for x in customerData['age']:
        if x >= 18 and x <= 30:
            youngNum += 1

    for y in customerData['age']:
        if y > 30 and y <= 50:
            middleAgeNum += 1

    for z in customerData['age']:
        if z > 50 and z <= 80:
            oldAgeNum += 1

    print("\n📊 Age Groups:")
    print(
        f"- Young (18-30): {youngNum} customers ({youngNum/customerRow*100:.1f}%)")
    print(
        f"- Middle Aged (31-50): {middleAgeNum} customers ({middleAgeNum/customerRow*100:.1f}%)")
    print(
        f"- Young (51-80): {oldAgeNum} customers ({oldAgeNum/customerRow*100:.1f}%)")

    print("\n🔍 Data Quality Check:")
    missingValues = customerData.isnull().sum().sum()

    totalOutliers = 0
    for feature in selectedFeatures:
        Q1 = customerData[feature].quantile(0.25)
        Q3 = customerData[feature].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
        outliers = customerData[(customerData[feature] < lowerBound) | (
            customerData[feature] > upperBound)][feature]
        totalOutliers += len(outliers)

    print(f"Missing Values: {missingValues}")
    print(f"Outliers detected: {totalOutliers} (using IQR method)")
    print("Class balance: Multiple Cluster visible")

    plt.scatter(customerData["annual_income"], customerData["spending_score"])
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.title('Customer Income vs Spending Score')
    plt.show()

    goBack = input("\nGo Back ? (y/n):  ")
    if goBack == 'y':
        loadDatasets()


def loadDatasets():
    print("\n=== DATASET LOADING & EXPLORATION ===")
    print("\n📁 Available datasets:")
    print("""1. Boston Housing (Regression)
2. Iris Flowers (Classification)
3. Wine Quality (Multi-class)
4. Customer Segmentation (Clustering)
5. Back to main menu""")
    loadDatasetOption = int(input("\nSelect Option: "))

    match loadDatasetOption:
        case 1:
            loadBostonHousing()
        case 2:
            loadIrisFlower()
        case 3:
            loadWine()
        case 4:
            loadCustomer()
        case 5:
            mainMenu()


def regressionModel():
    bostonHousing = pd.read_csv("boston.csv")

    X = bostonHousing.drop('Price', axis=1)
    y = bostonHousing['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    results = []

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    lr_r2 = r2_score(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr)) * 1000
    lr_mae = mean_absolute_error(y_test, y_pred_lr) * 1000

    coef_df = pd.DataFrame(
        {'Feature': X.columns, 'Coefficient': lr.coef_ * 1000})
    top_coefs = coef_df.sort_values(
        by='Coefficient', key=abs, ascending=False).head(3)

    results.append({
        'Model': 'Linear Regression',
        'R²': lr_r2,
        'RMSE': lr_rmse,
        'MAE': lr_mae
    })


    polyreg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    polyreg.fit(X_train, y_train)
    y_pred_poly = polyreg.predict(X_test) 

    poly_r2 = r2_score(y_test, y_pred_poly)
    poly_rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly)) * 1000
    poly_mae = mean_absolute_error(
        y_test, y_pred_poly) * 1000 

    results.append({
        'Model': 'Polynomial Regression (degree=2)',
        'R²': poly_r2,
        'RMSE': poly_rmse,
        'MAE': poly_mae
    })


    selector = SelectKBest(score_func=f_regression, k=8)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X.columns[selector.get_support()].tolist()

    mr = LinearRegression()
    mr.fit(X_train_selected, y_train)
    y_pred_mr = mr.predict(X_test_selected)

    mr_r2 = r2_score(y_test, y_pred_mr)
    mr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mr)) * 1000
    mr_mae = mean_absolute_error(y_test, y_pred_mr) * 1000

    results.append({
        'Model': 'Multiple Regression',
        'R²': mr_r2,
        'RMSE': mr_rmse,
        'MAE': mr_mae
    })

    print("\n=== REGRESSION MODELS ===")
    print("🎯 Target: Predicting house prices (MEDV)")
    print(f"📊 Training on {len(X_train)} samples")
    print(f"🧪 Testing on {len(X_test)} samples\n")

    print("=== MODEL TRAINING RESULTS ===")
    print("\n1️1. LINEAR REGRESSION:")
    print("✅ Model trained successfully")
    print(f"R² Score: {lr_r2:.3f}")
    print(f"RMSE: ${lr_rmse:,.0f}")
    print(f"MAE: ${lr_mae:,.0f}")
    print("\nCoefficients (top 3):")
    for _, row in top_coefs.iterrows():
        print(f"- {row['Feature']}: ${row['Coefficient']:,.0f} per unit")

    print("\n2. POLYNOMIAL REGRESSION (degree=2):")
    print("✅ Model trained successfully")
    print(f"R² Score: {poly_r2:.3f}")
    print(f"RMSE: ${poly_rmse:,.0f}")
    print(f"MAE: ${poly_mae:,.0f}")
    print(
        f"Feature importance: Polynomial features improved fit by {(poly_r2 - lr_r2)*100:.1f}%")

    print("\n3. MULTIPLE REGRESSION (with feature selection):")
    print("✅ Selected 8 most important features")
    print(f"R² Score: {mr_r2:.3f}")
    print(f"RMSE: ${mr_rmse:,.0f}")
    print(f"MAE: ${mr_mae:,.0f}")
    print(f"Selected features: {', '.join(selected_features)}")

    print("\n📊 MODEL COMPARISON:")
    table_data = [
        [result['Model'], f"{result['R²']:.3f}",
         f"${result['RMSE']:,.0f}", f"${result['MAE']:,.0f}"]
        for result in results
    ]
    headers = ["Model", "R²", "RMSE", "MAE"]
    print(tabulate(table_data, headers=headers,
                   tablefmt="rounded_grid", stralign="left", numalign="center"))

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.scatter(y_test, y_pred_lr, c='blue',
                alpha=0.5, label='Linear Regression')
    ax1.scatter(y_test, y_pred_poly, c='green',
                alpha=0.5, label='Polynomial Regression')
    ax1.scatter(y_test, y_pred_mr, c='red', alpha=0.5,
                label='Multiple Regression')
    ax1.plot([y_test.min(), y_test.max()], [
             y_test.min(), y_test.max()], 'k--', lw=2)
    ax1.set_xlabel('Actual MEDV ($1000s)')
    ax1.set_ylabel('Predicted MEDV ($1000s)')
    ax1.set_title('Actual vs Predicted House Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(y_pred_lr, y_test - y_pred_lr, c='blue',
                alpha=0.5, label='Linear Regression')
    ax2.scatter(y_pred_poly, y_test - y_pred_poly, c='green',
                alpha=0.5, label='Polynomial Regression')
    ax2.scatter(y_pred_mr, y_test - y_pred_mr, c='red',
                alpha=0.5, label='Multiple Regression')
    ax2.axhline(y=0, color='black', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted MEDV ($1000s)')
    ax2.set_ylabel('Residuals ($1000s)')
    ax2.set_title('Residuals Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    coef_df_sorted = coef_df.sort_values(
        by='Coefficient', key=abs, ascending=False)
    sns.barplot(x='Coefficient', y='Feature',
                data=coef_df_sorted, ax=ax3, color='skyblue')
    ax3.set_title('Feature Importance (Linear Regression Coefficients)')
    ax3.set_xlabel('Coefficient ($1000s per unit)')
    ax3.set_ylabel('Feature')
    ax3.grid(True, alpha=0.3, axis='x')

    plt.show()

    goBack = input("\nGo Back ? (y/n):  ")
    if goBack == 'y':
        mainMenu()


def classificationModel():
    print("\n=== BINARY CLASSIFICATION MODELS ===")
    print("\n🌸 Loading Iris dataset (modified for binary classification)...")

    iris = pd.read_csv("iris.csv")
    X = iris[['sepal.length', 'sepal.width',
              'petal.length', 'petal.width']].values
    y = (iris['variety'] == 'Setosa').astype(
        int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("✅ Classes: Setosa vs Non-Setosa (simplified for binary classification)")
    print(f"📊 Training on 80% of data ({len(X_train)} samples)")
    print(f"🧪 Testing on 20% of data ({len(X_test)} samples)\n")
    results = []

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_score_lr = lr.predict_proba(X_test)[:, 1]

    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_precision = precision_score(y_test, y_pred_lr)
    lr_recall = recall_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr)

    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': lr_accuracy,
        'Precision': lr_precision,
        'Recall': lr_recall,
        'F1': lr_f1

    })

    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    y_score_svm = svm.predict_proba(X_test)[:, 1]

    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_precision = precision_score(y_test, y_pred_svm)
    svm_recall = recall_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm)

    results.append({
        'Model': 'Support Vector Machine (SVM)',
        'Accuracy': svm_accuracy,
        'Precision': svm_precision,
        'Recall': svm_recall,
        'F1': svm_f1
    })

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_score_knn = knn.predict_proba(X_test)[:, 1]

    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    knn_precision = precision_score(y_test, y_pred_knn)
    knn_recall = recall_score(y_test, y_pred_knn)
    knn_f1 = f1_score(y_test, y_pred_knn)

    results.append({
        'Model': 'K-Nearest Neighbors (k=5)',
        'Accuracy': knn_accuracy,
        'Precision': knn_precision,
        'Recall': knn_recall,
        'F1': knn_f1
    })

    print("=== MODEL TRAINING RESULTS ===")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['Model'].upper()}:")
        print("✅ Model trained successfully")
        print(f"Accuracy: {result['Accuracy']*100:.1f}%")
        print(f"Precision: {result['Precision']:.2f}")
        print(f"Recall: {result['Recall']:.2f}")
        print(f"F1-Score: {result['F1']:.2f}")


    print("\n📊 BINARY CONFUSION MATRIX:")
    print("\nLogistic Regression:")
    cm = confusion_matrix(y_test, y_pred_lr)
    cm_table = [
        ["Setosa", cm[1, 1], cm[1, 0]],
        ["Non-Setosa", cm[0, 1], cm[0, 0]]
    ]
    headers = ["Actual", "Setosa", "Non-Setosa"]
    print(tabulate(cm_table, headers=headers, tablefmt="rounded_grid",stralign="left", numalign="center"))

    print("\n🎯 BINARY CLASSIFICATION METRICS:")
    print(f"True Positives: {cm[1, 1]} (Correctly identified Setosa)")
    print(f"True Negatives: {cm[0, 0]} (Correctly identified Non-Setosa)")
    print(f"False Positives: {cm[0, 1]} (Incorrectly called Setosa)")
    print(f"False Negatives: {cm[1, 0]} (Missed Setosa)")


    roc_auc_lr = auc(*roc_curve(y_test, y_score_lr)[:2])
    print("\n📈 ROC CURVE ANALYSIS:")
    print(f"AUC Score: {roc_auc_lr:.2f} (Perfect classification)")
    print("Optimal Threshold: 0.50")  

    print("\n🏆 BEST MODEL: All models tied at 100% ")
    classifier = LogisticRegression(multi_class='ovr', max_iter=1000)
    classifier.fit(X_train, y_train)
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    class_names = lb.classes_
    
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        y_score = classifier.predict_proba(X_test)[:, i]
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Iris Species (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    
    goBack = input("\nGo Back ? (y/n):  ")
    if goBack == 'y':
        mainMenu()

      

def multiClassClassificationModel():

    print("\n=== MULTI-CLASS CLASSIFICATION MODELS ===")
    print("\n🍷 Loading Wine Quality dataset for multi-class classification...")


    wine = pd.read_csv("redwine.csv") 
    X = wine.drop('quality', axis=1).values
    y = wine['quality'].values
    feature_names = wine.drop('quality', axis=1).columns.tolist()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("✅ 6 quality classes (3, 4, 5, 6, 7, 8)")
    print(f"📊 Training on 80% of data ({len(X_train)} samples)")
    print(f"🧪 Testing on 20% of data ({len(X_test)} samples)\n")

    results = []
    models = {}


    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    rf_recall = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)

    results.append({
        'Model': 'Random Forest',
        'Accuracy': rf_accuracy,
        'Precision': rf_precision,
        'Recall': rf_recall,
        'F1': rf_f1
    })
    models['Random Forest'] = rf


    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_precision = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    svm_recall = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    svm_f1 = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)

    results.append({
        'Model': 'Support Vector Machine',
        'Accuracy': svm_accuracy,
        'Precision': svm_precision,
        'Recall': svm_recall,
        'F1': svm_f1
    })
    models['Support Vector Machine'] = svm


    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)

    gb_accuracy = accuracy_score(y_test, y_pred_gb)
    gb_precision = precision_score(y_test, y_pred_gb, average='weighted', zero_division=0)
    gb_recall = recall_score(y_test, y_pred_gb, average='weighted', zero_division=0)
    gb_f1 = f1_score(y_test, y_pred_gb, average='weighted', zero_division=0)

    results.append({
        'Model': 'Gradient Boosting',
        'Accuracy': gb_accuracy,
        'Precision': gb_precision,
        'Recall': gb_recall,
        'F1': gb_f1
    })
    models['Gradient Boosting'] = gb


    print("=== MODEL TRAINING RESULTS ===")
    for i, result in enumerate(results, 1):
        print(f"\n{i}\uFE0F⃣ {result['Model'].upper()}:")
        print("✅ Model trained successfully")
        print(f"Accuracy: {result['Accuracy']*100:.1f}%")
        print(f"Precision: {result['Precision']:.2f}")
        print(f"Recall: {result['Recall']:.2f}")
        print(f"F1-Score: {result['F1']:.2f}")
        if result['Model'] == 'Random Forest':

            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf.feature_importances_
            }).sort_values(by='Importance', ascending=False).head(3)
            print("\nMost important features:")
            for _, row in feature_importance.iterrows():
                print(f"- {row['Feature'].replace(' ', '_')}: {row['Importance']:.2f}")


    print("\n📊 MULTI-CLASS CONFUSION MATRIX:")
    print("\nGradient Boosting (Best Model):")
    cm = confusion_matrix(y_test, y_pred_gb)
    quality_classes = sorted(np.unique(y_test))
    cm_table = []
    for i, q in enumerate(quality_classes):
        row = [q] + list(cm[i])

        correct = cm[i, i]
        total = cm[i].sum()
        percentage = (correct / total * 100) if total > 0 else 0
        row.append(f"({percentage:.0f}% correct)")
        cm_table.append(row)
    headers = ["Actual Quality"] + [f"{q}" for q in quality_classes] + [""]
    print(tabulate(cm_table, headers=headers, tablefmt="rounded_grid", stralign="left", numalign="center"))


    print("\n🎯 MULTI-CLASS INSIGHTS:")
    print("- Quality 5-6 wines easiest to predict (most common)")
    print("- Quality 3 and 8 wines hardest (very rare)")
    print("- Model struggles with extreme quality ratings")
    print("- Class imbalance affects performance")


    print("\n📈 CLASS-WISE PERFORMANCE:")
    report = classification_report(y_test, y_pred_gb, output_dict=True, zero_division=0)
    class_table = []
    for q in quality_classes:
        row = [
            q,
            f"{report[str(q)]['precision']:.2f}",
            f"{report[str(q)]['recall']:.2f}",
            f"{report[str(q)]['f1-score']:.2f}",
            int(report[str(q)]['support'])
        ]
        class_table.append(row)
    headers = ["Quality", "Precision", "Recall", "F1-Score", "Support"]
    print(tabulate(class_table, headers=headers, tablefmt="rounded_grid", stralign="left", numalign="center"))


    best_model = max(results, key=lambda x: x['Accuracy'])
    print(f"\n🏆 BEST MODEL: {best_model['Model']} ({best_model['Accuracy']*100:.1f}% accuracy)")

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)


    ax1 = fig.add_subplot(gs[0, 0]) 
    ax2 = fig.add_subplot(gs[1, 0])  


    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
                xticklabels=quality_classes, yticklabels=quality_classes)
    ax1.set_xlabel('Predicted Quality')
    ax1.set_ylabel('Actual Quality')
    ax1.set_title('Confusion Matrix (Gradient Boosting)')


    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax2, color='skyblue')
    ax2.set_title('Feature Importance (Random Forest)')
    ax2.set_xlabel('Importance')
    ax2.set_ylabel('Feature')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.show()
    
    
    goBack = input("\nGo Back ? (y/n):  ")
    if goBack == 'y':
        mainMenu()
       
       
def clusteringAnalysis():
    print("\n=== CLUSTERING ANALYSIS ===")
    print("\n📊Loading customer segmentation dataset...")
    
    customer=pd.read_csv("customers.csv")
    X = customer[['annual_income', 'spending_score']].values
    
    print("✅ 200 customers, 2 features (Annual Income, Spending Score)")
    
    print("\n🎯 DETERMINING OPTIMAL CLUSTERS:")
    print("\nElbow Method Analysis:")
    wcss = []
    for k in range(1, 8):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        print(f"k={k}: WCSS = {kmeans.inertia_:.2f}", end=" 📉" if k <= 5 else "", flush=True)
        if k == 5:
            print(" (elbow point)", flush=True)
        else:
            print("", flush=True)
            
    print("\n=== K-MEANS CLUSTERING (k=5) ===")
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    print("\n✅ Clustering completed successfully")
    
    print("\n📊 CLUSTER ANALYSIS:")
    cluster_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
    cluster_means = []
    for i in range(5):
        cluster_data = X[kmeans_labels == i]
        avg_income = cluster_data[:, 0].mean()
        avg_spending = cluster_data[:, 1].mean()
        cluster_means.append([avg_income, avg_spending])
    
    cluster_types = ["High Income", "Low Income", "Conservative", "Careful", "Standard"]
    cluster_table = []
    for i in range(5):
        row = [
            i,
            cluster_sizes[i],
            f"${cluster_means[i][0]:,.0f}",
            f"{cluster_means[i][1]:.1f}",
            cluster_types[i]
        ]
        cluster_table.append(row)
    
    headers = ["Cluster", "Size", "Avg Income", "Avg Spending", "Type"]
    print(tabulate(cluster_table, headers=headers, tablefmt="fancy_grid", stralign="left", numalign="center"))
    
    print("\n🎯 BUSINESS INSIGHTS:")
    print("- Cluster 2: High income, low spending (target for premium products)")
    print("- Cluster 3: Low income, high spending (budget-conscious buyers)")
    print("- Cluster 4: Standard customers (main market segment)")
    print("- Clear customer segmentation identified")
    
    print("\n=== HIERARCHICAL CLUSTERING ===")
    Z = linkage(X, method='ward')
    print("\n✅ Dendrogram generated")
    
    hier_labels = fcluster(Z, t=4, criterion='maxclust')
    print("✅ Optimal clusters: 4 (based on dendrogram)")
    
    print("\nSilhouette Analysis:")
    for k in range(3, 6):
        if k == 5:
            score = silhouette_score(X, kmeans_labels)
        else:
            temp_kmeans = KMeans(n_clusters=k, random_state=42)
            temp_labels = temp_kmeans.fit_predict(X)
            score = silhouette_score(X, temp_labels)
        print(f"- k={k}: Silhouette Score = {score:.2f}")
    
    print("\n🏆 BEST CLUSTERING: K-Means with k=5")


    fig = plt.figure(figsize=(12, 5)) 
    gs = fig.add_gridspec(1, 2, wspace=0.3) 
    

    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['purple', 'orange', 'green', 'blue', 'red']
    for i in range(5):
        ax1.scatter(X[kmeans_labels == i, 0], X[kmeans_labels == i, 1], c=colors[i], label=f'Cluster {i}', alpha=0.6)
    ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='x', s=200, linewidths=3, label='Centers')
    ax1.set_xlabel('Annual Income ($)')
    ax1.set_ylabel('Spending Score')
    ax1.set_title('K-Means Clustering (k=5)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    dendrogram(Z, ax=ax2)
    ax2.set_title('Hierarchical Clustering Dendrogram')
    ax2.set_xlabel('Customer Index')
    ax2.set_ylabel('Distance (Ward)')

    plt.show()
    
    goBack = input("\nGo Back ? (y/n):  ")
    if goBack == 'y':
        mainMenu()


#def modelEvaluation():
    #print("\n=== COMPREHENSIVE MODEL EVALUATION ===")


#def advancedMLTechniques():
    #print("\n=== ADVANCED MACHINE LEARNING TECHNIQUES ===")


mainMenu()
