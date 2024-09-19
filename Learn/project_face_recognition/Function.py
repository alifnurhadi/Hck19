import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


from sklearn.preprocessing import OrdinalEncoder
from feature_engine.outliers import Winsorizer
from sklearn.metrics import classification_report
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def elbow_method( scaled_pca: np.ndarray , maxcluster:int, randomstate:int=42):
    wcss = []
    random_state = randomstate
    max_cluster = maxcluster
    for i in range(2, max_cluster+1):
        km = KMeans(n_clusters = i,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = random_state)
        km.fit(scaled_pca)
        wcss.append(km.inertia_)

    plt.plot(range(2, max_cluster+1), wcss, marker ="o")
    plt.grid()
    plt.title('Elbow Method', fontsize = 20)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def plot_by_silhouette(range_n_clusters:list, X:np.ndarray, random_state:int=42):
    '''range_n_clusters is basicly a range of number cluster that want to be iterate or compare
     example [1,2,3,4,5,6,7] , or make it from range() syntax  '''
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 4)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters = n_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = random_state)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

def simplified_withregex(df: pd.DataFrame, column_name: str, patterns: dict):
    '''
    ### MAKE SURE ALREADY IMPORT REGEX by 'import re' at the very top of ur code.

    how to use : 

    df_result = simplified_withregex(df_data2, 'source', patterns1)
    df_result

    expected result of add a new dataframe at the end of column

    Dict example:

    patterns = {
    'google': r'(?:^.+[\w+\d\.]+)?google',
    'youtube': r'(?:^.+[\w+\d\.]+)?youtube',
    'facebook': r'(?:^.+[\w+\d\.]+)?facebook',
    'reddit': r'(?:^.+[\w+\d\.]+)?reddit',
    'yahoo': r'(?:^.+[\w+\d\.]+)?yahoo',
    'bing': r'(?:^.+[\w+\d\.]+)?bing',
    'baidu': r'(?:^.+[\w+\d\.]+)?baidu',
    'other': r'^(?!.*(?:google|youtube|facebook|reddit|yahoo|bing|baidu)).*$'
        }
    
    to debug or wanna try for a single web will work or not, try :
    
    test_input = 'www.analytic.duduc.com'

    for name, pattern in patterns1.items():
    match = re.search(pattern, test_input)
    print(f"Pattern '{name}' found in '{test_input}': {bool(match)}")

    column_name is the name of the column containing the URLs to be matched.

    '''
    
    def find_matching_patterns(url: str, patterns: dict):
        for key, pattern in patterns.items():
            if re.search(pattern, str(url), re.IGNORECASE):
                return key
        return 'Unrecognized'

    results = []
    for value in df[column_name]:
        if pd.isna(value):
            results.append(value)  # Keep NaN/null values as is
        else:
            results.append(find_matching_patterns(value, patterns))

    df[f'new_{column_name}'] = results
    return df


def regiongrouping_cardin(df:pd.DataFrame,column_name:str):
    ''' use it for only if the unique value are a list of countries,
    or u'll regret'''
    
    REGION_DICT = {
    "North America": [
        "United States", "Canada", "Mexico", "Puerto Rico", "Bermuda", 
        "Dominican Republic", "Honduras", "Costa Rica", "Panama", "El Salvador", 
        "Guatemala", "Nicaragua", "Jamaica", "Haiti", "Cuba", "Bahamas", 
        "Trinidad & Tobago", "Barbados", "Grenada", "Saint Lucia", 
        "Saint Vincent & Grenadines", "Antigua & Barbuda", "Saint Kitts & Nevis", 
        "Belize", "Aruba", "Curaçao", "Sint Maarten", "Anguilla", "Montserrat", 
        "British Virgin Islands", "Caribbean Netherlands", "U.S. Virgin Islands", 
        "Turks & Caicos Islands", "Cayman Islands", "Greenland", 
        "St. Pierre & Miquelon"
    ],
    "South America": [
        "Brazil", "Argentina", "Colombia", "Peru", "Venezuela", "Chile", 
        "Ecuador", "Bolivia", "Uruguay", "Paraguay", "Guyana", "Suriname", 
        "French Guiana"
    ],
    "Western Europe": [
        "United Kingdom", "Germany", "France", "Spain", "Italy", "Netherlands", 
        "Belgium", "Switzerland", "Austria", "Ireland", "Portugal", "Greece", 
        "Luxembourg", "Malta", "Monaco", "Liechtenstein", "Andorra", "San Marino", 
        "Vatican City", "Isle of Man", "Gibraltar", "Guernsey", "Jersey", 
        "Sweden", "Norway", "Denmark", "Finland", "Iceland", "Cyprus","Czech Republic"
    ],
    "Eastern Europe": [
        "Russia", "Poland", "Ukraine", "Romania", "Belarus", "Czechia", "Hungary", 
        "Bulgaria", "Slovakia", "Croatia", "Serbia", "Lithuania", "Latvia", 
        "Estonia", "Slovenia", "Bosnia & Herzegovina","Bosnia and Herzegovina", "Montenegro", 
        "Macedonia (FYROM)", "Kosovo", "Moldova", "Georgia", "Armenia", 
        "Azerbaijan", "Albania"
    ],
    "Asia": [
        "China", "India", "Japan", "South Korea", "Pakistan", 
        "Bangladesh", "Hong Kong", "Taiwan", "Sri Lanka", "Nepal", "Kazakhstan", "Uzbekistan", "Turkmenistan", 
        "Kyrgyzstan", "Tajikistan", "Afghanistan", "Mongolia", "Maldives", 
        "Bhutan", "Macau"
    ],
    "S-E-A": [
        "Indonesia","Myanmar","Vietnam", "Thailand", "Malaysia", "Singapore", 
        "Philippines", "Myanmar (Burma)", "Brunei", "Laos", "Cambodia", "Timor-Leste",
    ],
    "Middle East": [
        "Israel", "Saudi Arabia", "United Arab Emirates", "Turkey", "Iran", 
        "Egypt", "Jordan", "Lebanon", "Kuwait", "Qatar", "Oman", "Bahrain", 
        "Syria", "Iraq", "Palestine", "Yemen", "Libya"
    ],
    "Africa": [
        "South Africa", "Nigeria", "Kenya", "Ghana", "Morocco", "Algeria", 
        "Tunisia", "Ethiopia", "Tanzania", "Uganda", "Cameroon", "Ivory Coast (Côte d’Ivoire)", 
        "Senegal", "Mali", "Botswana", "Zambia", "Mozambique", "Namibia", 
        "Madagascar", "Angola", "Zimbabwe", "Mauritius", "Sudan", "South Sudan", 
        "Rwanda", "Burundi", "Togo", "Benin", "Central African Republic", 
        "Niger", "Chad", "Sierra Leone", "Liberia", "Guinea", "Equatorial Guinea", 
        "Gabon", "Congo - Brazzaville", "Congo - Kinshasa", "Comoros", 
        "Djibouti", "Somalia", "Eritrea", "Seychelles", "Cape Verde", 
        "Sao Tome & Principe", "Lesotho", "Eswatini (Swaziland)", 
        "Guinea-Bissau", "Gambia", "Mayotte", "Reunion", "Mauritania", "Malawi", 
        "Burkina Faso"
    ],
    "Oceania": [
        "Australia", "New Zealand", "Fiji", "Papua New Guinea", "Vanuatu", 
        "Solomon Islands", "Micronesia", "Palau", "Marshall Islands", "Tonga", 
        "Samoa", "Kiribati", "Tuvalu", "Nauru", "New Caledonia", 
        "French Polynesia", "Wallis & Futuna", "Pitcairn Islands", 
        "Norfolk Island", "Cook Islands", "Niue", "Tokelau", "American Samoa", 
        "Northern Mariana Islands", "Guam", "Samoa", "Micronesia"
    ],
    "Caribbean": [
        "Puerto Rico", "Dominican Republic", "Jamaica", "Trinidad & Tobago", 
        "Bahamas", "Barbados", "Saint Lucia", "Grenada", "Saint Vincent & Grenadines", 
        "Antigua & Barbuda", "Saint Kitts & Nevis", "Cayman Islands", 
        "Turks & Caicos Islands", "Bermuda", "Aruba", "Curaçao", "Sint Maarten", 
        "Caribbean Netherlands", "U.S. Virgin Islands", "Montserrat", "Anguilla", 
        "British Virgin Islands", "Dominica", "St. Martin"
    ],
    "Other": [
        "(not set)", "Greenland", "Faroe Islands", "Svalbard & Jan Mayen", 
        "Saint Pierre & Miquelon", "Falkland Islands", "Bouvet Island", 
        "French Southern Territories", "Heard Island & McDonald Islands", 
        "South Georgia & South Sandwich Islands", "British Indian Ocean Territory", 
        "Christmas Island", "Cocos (Keeling) Islands", "Norfolk Island", 
        "Pitcairn Islands", "Tokelau", "Niue", "Cook Islands", 
        "Wallis & Futuna", "French Polynesia", "New Caledonia", "Saint Martin", 
        "São Tomé & Principe",'Côte d’Ivoire', 'Réunion', 'St. Lucia', 'Swaziland', 'Guadeloupe',
       'Martinique', 'St. Kitts & Nevis', 'St. Vincent & Grenadines',
       'São Tomé & Príncipe'
    ]
}

    regions = []

    for value in df[column_name]:
        if value == None:
            regions.append(np.nan)
            continue
        else:
            for region,country in REGION_DICT.items():
                if value in country:
                    regions.append(region)
                    break
            else:
                regions.append('Other')


    df[f'''region_{column_name}'''] = regions
    return df

def freqgrouping_cardin(df:pd.DataFrame,column_name:str):

    group = []

    the_top_d = df[column_name].value_counts().sort_values(ascending=False).to_dict()
    top_list = list(the_top_d)

    for value in df[column_name]:
        if value == top_list[0]:
            group.append(f'''1st of {column_name}''')
        elif value in top_list[1:6] :
            group.append(f'''2nd - 5th of {column_name}''')
        elif value in top_list[6:11] :
            group.append(f'''6th - 10th of {column_name}''')
        elif value in top_list[11:21] : 
            group.append(f'''11th - 20th of {column_name}''')
        else:
            group.append(f'regular_{column_name}')

    df[f'''most_freq_{column_name}'''] = group
    
    return df



def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    ''' 
    use the latest pre-proccessing dataframe!!!
    \n model need to define first and set into variable.
    '''
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def DistributionCheck(table:pd.DataFrame,*column_name:list):
    ''' for dataframe data, please initialize first by read the document
        this can be done after u've done with pd.read actions.'''
    try:
        data= pd.DataFrame()
        kolom = []
        skew = []
        level = []
        outlier = []

        for column in column_name:
            skewness = table[column].skew()
            if -0.5 <= skewness <= 0.5:
                kolom.append(column)
                skew.append(skewness)
                level.append('Normal')
                
                avg = table[column].mean()
                std = table[column].std()

                up_bound = avg + 3*std
                low_bound = avg - 3*std

                outlier_value = table[(table[column]<low_bound) | (table[column]>up_bound)]
                value = outlier_value[column].unique().tolist()
                outlier.append(value)

            elif skewness<-1 or skewness>1 :
                kolom.append(column)
                skew.append(skewness)
                level.append('Ekstrim')

                q1 = table[column].quantile(0.25)
                q3 = table[column].quantile(0.75)
                iqr = q3-q1

                up_bound = q3 + 3*iqr
                low_bound = q1 - 3*iqr

                outlier_value = table[(table[column]<low_bound) | (table[column]>up_bound)]
                value = outlier_value[column].unique().tolist()
                outlier.append(value)

            else:
                kolom.append(column)
                skew.append(skewness)
                level.append('Moderate')

                q1 = table[column].quantile(0.25)
                q3 = table[column].quantile(0.75)
                iqr = q3-q1

                up_bound = q3 + 1.5*iqr
                low_bound = q1 - 1.5*iqr

                outlier_value = table[(table[column]<low_bound) | (table[column]>up_bound)]
                value = outlier_value[column].unique().tolist()
                outlier.append(value)

        data = pd.DataFrame({
            'Nama-kolom': kolom ,
            'Nilai-Skewness': skew ,
            'Status': level,
            'Nilai-outlier' : outlier 
        })
        return data
    
    except ValueError :
        return f'ValueError: Can"t check the distribution, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t check the distribution, please check data-type needed for its parameter' 

def encode_categorical(df:pd.DataFrame, column:str, categorical_order:list):
    ''' set the categorical order from lowest to highest and save to a variable to be called in this function 
        e.g 
        countryrank_order = ['regular_country', '11th - 20th of country', '6th - 10th of country','2nd - 5th of country','1st of country']

        prepare a variable to saved this process or u can overwrite the existing one
        '''
    if len(categorical_order) != len(set(categorical_order)):
        raise ValueError("categorical_order must have unique values")

    df[column] = pd.Categorical(df[column], categories=categorical_order, ordered=True).codes
    return df


def vis_FeatureImportance (model, df1:pd.DataFrame, df2:pd.DataFrame, args:list=None ):
    ''' USED THIS TO A SUPERVISED LEARNING ONLY

        model = tree-base function that will be used
        \n df1 = xtrain data
        \n df2 = ytrain data that already encoded
        \n * args = list of xtrain columns that need to be one hot encoded if already encoded just put None inside the argument 
        \n IF THERES A CATEGORICAL COLUMNS, IT NEED TO BE ENCODED OUTSIDE THIS AND BEFORE USING THIS!!!
        '''
    
    if args is not None:
        X_train_ohe = pd.get_dummies(df1, prefix='', prefix_sep='', columns=args,) # its a column that need to be encoded
    else :
        X_train_ohe = df1.copy()

    y_train_ohe = df2.copy()
    model.fit(X_train_ohe, y_train_ohe)

    feat_importances = pd.Series(model.feature_importances_, index=X_train_ohe.columns)
    feat_importances.nlargest(10).plot(kind='barh').invert_yaxis()    
    plt.title('Top 10 Feature Importances')
    plt.show()

def vif_analysis():
    print('''
          from statsmodels.stats.outliers_influence import variance_inflation_factor
          
          def vif_analysis(origin_pd, target_column):
        # Remove the target variable from the feature set
        features = origin_pd.drop(columns=[target_column])

        # Ensure all features are numeric
        features = features.select_dtypes(include=[np.number])

        # Handle missing values (e.g., drop rows with NaNs)
        features = features.dropna()

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features.columns
        vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

        # Sort VIF values in descending order
        vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

        return vif_data



    vif_data = vif_analysis(origin_pd, 'Exited')

    recommendations = []
    for _, row in vif_data.iterrows():
        if row["VIF"] > 10:
            recommendations.append(f"High multicollinearity: Consider removing or combining {row['Feature']}")
        elif 5 < row["VIF"] <= 10:
            recommendations.append(f"Moderate multicollinearity: Monitor {row['Feature']}")
        else:
            recommendations.append(f"Low multicollinearity: Keep {row['Feature']}")

    vif_data["Recommendation"] = recommendations

    vif_data
    ''')

def check_stationarity(series):

    result = adfuller(series.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

def find_best_model(data:pd.DataFrame, num_p:int, num_d:int, num_q:int):
  ''' write the p d q number according to pacf acf plot and after checking stationary'''
  df = pd.DataFrame() #We make a pandas dataframe to ease our work, you can use any way that makes you comfortable and work easly
  pdq = [[],[],[]] #This list for storing the combinations of p and q
  aic = [] # This list for storing aic value
  for p in range(num_p + 1): #plus one is to make the range up to num_p since python range only ranging from 0 to N-1
    for d in range(num_d + 1):
      for q in range(num_q + 1):
        # #if p!=0 and q!=0: #this logic will avoid (0,0) value which is not correct
        try: #Using exception to avoid the error while training the model with certain p and q value
          model = ARIMA(data, order=(p, d, q))
          result = model.fit()
          pdq[0].append(p)
          pdq[1].append(d)
          pdq[2].append(q)
          aic.append(result.aic)
        except:
          pass #If there is an error caused by the calculation, it will be skipped to the next iteration

  df['p'] = pdq[0]
  df['d'] = pdq[1]
  df['q'] = pdq[2]
  df['AIC'] = aic

  df.sort_values('AIC', inplace=True)

  return df.p.iloc[0], df.d.iloc[0], df.q.iloc[0], df.AIC.iloc[0], df

def select_important_features(model, df1:pd.DataFrame, df2:pd.DataFrame, args:list=None , threshold=0.95):
    """
    Select features that contribute to the cumulative importance up to the given threshold.
    
    :param feat_importances: Series containing feature importance scores with feature names as the index.
    :param threshold: Cumulative importance threshold (default is 0.95 for 95%).
    
    :return: Index of selected features.
    """
    if args is not None:
        X_train_ohe = pd.get_dummies(df1, prefix='', prefix_sep='', columns=args) # its a column that need to be encoded
    else :
        X_train_ohe = df1.copy()

    y_train_ohe = df2.copy()
    model.fit(X_train_ohe, y_train_ohe)

    feat_importances = pd.Series(model.feature_importances_, index=X_train_ohe.columns)
    cumulative_importance = feat_importances.cumsum()
    selected_features = cumulative_importance[cumulative_importance <= threshold].index
    return selected_features


def help_featureimportance():
    ''' use when all the data are encoded well enough and use this as reference'''

    print('''
    X_eda = df_data2[['bounces', 'time_on_site', 'pageviews']]
    y_eda = df_data2['will_buy_on_return_visit']

    X_train, X_test, y_train, y_test = train_test_split(X_eda, y_eda, test_size=0.2, random_state=25)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    importance = pd.DataFrame({'feature': X_eda.columns, 'importance': rf.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    print(importance)

    plt.figure(figsize=(10,6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title('Feature Importance')
    plt.show()''')

def FeatureImportance_list (model, df1:pd.DataFrame, df2:pd.DataFrame, args:list=None, treshold:float = 0.90):
    ''' USED THIS TO A SUPERVISED LEARNING ONLY

        model = tree-base function that will be used
        \n df1 = xtrain data
        \n df2 = ytrain data that already encoded
        \n * args = list of xtrain columns that need to be one hot encoded if already encoded just put None inside the argument 
        \n IF THERES A CATEGORICAL COLUMNS, IT NEED TO BE ENCODED OUTSIDE THIS AND BEFORE USING THIS!!!
        '''
    
    X_train_processed = df1.copy()
    
    if args is not None:
        encoder = OrdinalEncoder()
        X_train_processed[args] = encoder.fit_transform(df1[args])
        
    y_train_ohe = df2.copy()
    model.fit(X_train_processed, y_train_ohe)

    feat_importances = pd.Series(model.feature_importances_, index=X_train_processed.columns)
    feat_importances = feat_importances.sort_values(ascending=False)
    cumulative_importance = feat_importances.cumsum()

    return cumulative_importance[cumulative_importance <= treshold]

def help_handling_outlier():
    return print(f''' ================|| do from feature_engine.outliers import Winsorizer at the very top ||================||
    winsoriser = Winsorizer(capping_method='iqr', # for normal dist, type 'gaussian': if still outlier out there put quantiles.
                            tail='both',
                            fold=       #3 for normal dist; 1.5 for moderate skew and 3 for extreme; 0.1 for quantiles.
                            variables= #[list of outlier columns that need to be handle],
                            missing_values='ignore')
    
    X_train_capped = winsoriser.fit_transform(X_train) # when using unsupervised only this line is needed.
    X_test_capped = winsoriser.transform(X_test)

    return X_train_capped, X_test_capped  when doing unsupervised, the return variable value is only one( which is the whole data )''')


def Correlation_test1(table:pd.DataFrame,maincolumn_name:str,alist:list,**kwargscolumn:dict):
    ''' for dataframe data, please initialize first by read the document'''
    try:
        www = DistributionCheck(table,*alist)
        for kol in www['status']:
            if ... == ... :
                
                corr_r, pval_p = stats.pearsonr(table[maincolumn_name], table[kol])
                
                if corr_r > 0.5:
                    print(f'kolom {maincolumn_name} ber-korelasi kuat dengan kolom {kol} dan berjalan linear.')
                elif corr_r > 0 :
                    print(f'kolom {maincolumn_name} ber-korelasi lemah hingga moderat dengan kolom {kol} dan berjalan linear.')
                elif corr_r < 0 :
                    print(f'kolom {maincolumn_name} ber-korelasi lemah hingga moderat dengan kolom {kol} dan berjalan berlawanan')
                elif corr_r < -0.5 :
                    print(f'kolom {maincolumn_name} ber-korelasi kuat dengan kolom {kol} dan berjalan berlawanan')
                else:
                    print(f'kolom {maincolumn_name} tidak ada korelasi dengan kolom {kol}')

    except ValueError :
        return f'ValueError: Can"t check the correlations, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t check the correlations, please check data-type needed for its parameter'  


def Correlation_test2(dataframe:pd.DataFrame,predik:str='target',*allcol,**korelasi:dict):
    ''' dataframe == please initialize first by read the document.
        predik == a target column save into some variable first.
        kwargs.value == fill with numeric column'''
    try:
        korelasii = []
        kolomfitur = []
        # allcol1 = dataframe.select_dtypes(include=np.number)
        # dataframe[allcol1].apply(lambda x: (x - x.mean()) / x.std())
        for nu in allcol:
            skewnes = dataframe[nu].skew()
            if nu in korelasi.values():
                
                if skewnes<= -0.5 or skewnes>= 0.5:
                    corr_rho, pval_s = stats.spearmanr(dataframe[predik], dataframe[nu])
                    if corr_rho > 0.5:
                        kolomfitur.append(nu)
                        korelasii.append('kuat')
                    elif corr_rho > 0 :
                        kolomfitur.append(nu)
                        korelasii.append('lemah')
                    elif corr_rho < 0 :
                        kolomfitur.append(nu)
                        korelasii.append('lemah_kontra')
                    elif corr_rho < -0.5 :
                        kolomfitur.append(nu)
                        korelasii.append('kuat_kontra')
                    else:
                        kolomfitur.append(nu)
                        korelasii.append('none')

            elif nu in korelasi.values():

                if -0.5 < skewnes < 0.5:
                    corr_r, pval_p = stats.pearsonr(dataframe[predik], dataframe[nu])
                    if corr_r > 0.5:
                        kolomfitur.append(nu)
                        korelasii.append('kuat')
                    elif corr_r > 0 :
                        kolomfitur.append(nu)
                        korelasii.append('lemah')
                    elif corr_r < 0 :
                        kolomfitur.append(nu)
                        korelasii.append('lemah_kontra')
                    elif corr_r < -0.5 :
                        kolomfitur.append(nu)
                        korelasii.append('kuat_kontra')
                    else:
                        kolomfitur.append(nu)
                        korelasii.append('none')
                    
            else:
                corr_tau, pval_k = stats.kendalltau(dataframe[predik], dataframe[nu])
                if corr_tau > 0.5:
                    kolomfitur.append(nu)
                    korelasii.append('kuat')
                elif corr_tau > 0 :
                    kolomfitur.append(nu)
                    korelasii.append('lemah')
                elif corr_tau < 0 :
                    kolomfitur.append(nu)
                    korelasii.append('lemah_kontra')
                elif corr_tau < -0.5 :
                    kolomfitur.append(nu)
                    korelasii.append('kuat_kontra')
                else:
                    kolomfitur.append(nu)
                    korelasii.append('none')

        data1 = pd.DataFrame()
        data1 = pd.DataFrame({
            'fitur Vs target': kolomfitur ,
            'korelasi': korelasii
        })
        return data1

    except ValueError :
        return f'ValueError: Can"t check the correlations, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t check the correlations, please check data-type needed for its parameter'
    
def corr_chisquared(table: pd.DataFrame, maincolumn_name: str, columnname: str):
    """
    Performs correlation analysis using Chi-Squared test between a main column and multiple other categorical columns in a DataFrame.
    
    Parameters:
    table (pd.DataFrame): The DataFrame containing the data.
    maincolumn_name (str): The name of the main categorical column to be analyzed.
    *columnname (list): The names of the other categorical columns to be analyzed.
    
    Returns:
    None
    """
    try:
        if table[maincolumn_name].dtype == 'object' and table[columnname].dtype == 'object':
            contingency_table = pd.crosstab(table[maincolumn_name], table[columnname])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            if p_value < 0.05:
                print(f"kolom {maincolumn_name} memiliki korelasi signifikan dengan {columnname}.")
            else:
                print(f"kolom {maincolumn_name} tidak-memiliki korelasi signifikan dengan {columnname} . ")
        else:
                print(f"Cannot perform Chi-Squared correlation analysis between {maincolumn_name} and {columnname} due to incompatible data types.")
    except ValueError:
        print("ValueError: Cannot perform Chi-Squared correlation analysis. Please check the data types of the columns.")
    except KeyError:
        print("KeyError: One or more columns not found in the DataFrame. Please check the column names.")

def central_tendency(table:pd.DataFrame,)->pd.DataFrame:
    ''' for dataframe data, please initialize first by read the document'''
    return table.describe()

def cardin_check(urdataframe:pd.DataFrame , *namakolom:list):
    ''' for dataframe data in train_set, please initialize first by read the document'''
    try:
        lis1=[]
        lis2=[]
        lis3=[]

        for kolom in namakolom:

            lis1.append(kolom)

            lis2.append(urdataframe[kolom].nunique())

            isi_kolom = list(urdataframe[kolom].unique())
            lis3.append(isi_kolom)

        df0 = pd.DataFrame({
            'nama-kolom' : lis1,
            'jumlah_unique' : lis2,
            'isi_unique' : lis3
            })
        return df0
    
    except ValueError :
        return f'ValueError: Can"t shows the cardinality, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t shows the cardinality, please check data-type needed for its parameter'
    

def vis_boxplot(tablename:pd.DataFrame,column_name:str):
    ''' for dataframe data, please initialize first by read the document'''
    try:
        q1 = tablename[column_name].quantile(0.25)
        q2 = tablename[column_name].quantile(0.5)
        q3 = tablename[column_name].quantile(0.75)

        iqr = q3 - q1
        plt.figure(figsize=(6, 4))
        plt.boxplot(tablename[column_name], vert=False, patch_artist=True)

        plt.annotate(f'Q1 = {q1}', xy=(q1, 1.06), xytext=(q1, 1.2), arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.annotate(f'Q2 = {q2}', xy=(q2, 1.06), xytext=(q2, 1.3), arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.annotate(f'Q3 = {q3}', xy=(q3, 1.06), xytext=(q3, 1.2), arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.annotate(f'IQR = {iqr}', xy=((q1+q3)/2, 0.85), ha='center')

        plt.axvline(q1, linestyle='--', color='gray', label='Q1')
        plt.axvline(q3, linestyle='--', color='gray', label='Q3')
        plt.axvspan(q1, q3, alpha=0.2, color='gray', label='IQR')

        plt.xlabel('Values')
        plt.title(f'Boxplot of {column_name}')

        return plt.show()
    except ValueError :
        return f'ValueError: Can"t shows the boxplot, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t shows the boxplot, please check data-type needed for its parameter'
    
def vis_barchart(tablename:pd.DataFrame , columnname:str):
    ''' for dataframe data, please initialize first by read the document'''
    try :
        tablename[columnname].value_counts().sort_index().plot(kind='bar')
    except ValueError :
        return f'ValueError: Can"t shows the barchart, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t shows the barchart, please check data-type needed for its parameter'
    
def vis_histogram(tablename:pd.DataFrame):
    ''' for dataframe data, please initialize first by read the document'''
    try:
        tablename.hist(figsize=(12,12))
    except ValueError :
        return f'ValueError: Can"t shows the histogram, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t shows the histogram, please check data-type needed for its parameter'
    
def vis_scatter_relation(table:pd.DataFrame, column1name:str, column2name:str):
    ''' for dataframe data, please initialize first by read the document'''
    try:
        sns.lmplot(data=table,x=column1name, y=column2name)
    except ValueError :
        return f'ValueError: Can"t shows the scatter relation plot, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t shows the scatter relation plot, please check data-type needed for its parameter'
    
def visualizeoutlier_normal(tablename:pd.DataFrame, columnname:str):
    ''' for dataframe data, please initialize first by read the document'''
    try: 
        avg = tablename[columnname].mean()
        std = tablename[columnname].std()

        up_bound = avg + 3*std
        low_bound = avg - 3*std
    
        print(f'''batas atas : {up_bound} \n batas bawah : {low_bound}''')
        
        fig, ax = plt.subplots(ncols=2,figsize=(10,4))
        tablename[columnname].plot(kind='hist',bins=20,ax=ax[0])
        tablename[columnname].plot(kind='box',ax=ax[1])

    except ValueError :
        return f'ValueError: Can"t shows the outlier, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t shows the outlier, please check data-type needed for its parameter'
    

def visualizeoutlier_skew(table:pd.DataFrame, columnname:str, number:float = 1.5):
    ''' for dataframe data, please initialize first by read the document'''
    try:
        q1 = table[columnname].quantile(0.25)
        q3 = table[columnname].quantile(0.75)
        iqr = q3-q1

        up_fence = q3 + number*iqr
        low_fence = q1 - number*iqr
        print(f'''batas atas : {up_fence} \n batas bawah : {low_fence}''')
        
        fig, ax = plt.subplots(ncols=2,figsize=(10,4))
        table[columnname].plot(kind='hist',bins=20,ax=ax[0])
        table[columnname].plot(kind='box',ax=ax[1])

    except ValueError :
        return f'ValueError: Can"t shows the outlier, Please check data-type needed for its parameter'
    except KeyError:
        return f'KeyError : Can"t shows the outlier, please check data-type needed for its parameter'

if __name__ == '__main__':
    print(''' ======================================================================================
            Please try to type any data manipulation you want to, 
            try if theres a suggestion then read the documentaion of it.
 ======================================================================================
          ''')