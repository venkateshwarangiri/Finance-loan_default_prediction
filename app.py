import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from xgboost import XGBClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)

icon = Image.open('apple-xxl.png')
st.set_page_config(page_title="LOAN DEFAULT PREDICTION",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded",
                   )
st.markdown("<h1 style='text-align: center; color: #051937;background-color:white;border-radius:15px;'>LOAN DEFAULT PREDICTION</h1>",
            unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu(None, ["ANALYSIS", "PREDICTION"],
                           icons=["bi bi-clipboard-data", "bi bi-magic"],
                           default_index=0,
                           orientation="vertical",
                           styles={"nav-link": {"font-size": "20px", "text-align": "centre", "margin-top": "20px",
                                                "--hover-color": "#266c81"},
                                   "icon": {"font-size": "20px"},
                                   "container": {"max-width": "6000px"},
                                   "nav-link-selected": {"background-color": "#266c81"}, })

# setting the back-ground color


def back_ground():
    st.markdown(f""" <style>.stApp {{
                        background-image: linear-gradient(to right top, #051937, #051937, #051937, #051937, #051937);;
                        background-size: cover}}
                     </style>""", unsafe_allow_html=True)


back_ground()


if selected == "ANALYSIS":
    st.markdown("")
    st.markdown("## :white[SAMPLE DATA]")
    df_s = pd.read_csv('loan_default_prediction_project(1).csv')
    st.table(df_s.head())

    st.markdown("### :white[FEATURE WISE ANALYSIS]")
    st.markdown("")
    st.markdown("")
    with st.spinner("please wait..."):
        st.pyplot(sns.pairplot(df_s))

    st.markdown(
        "#### :white[FROM THE PLOT WE CAN INFERE THERE IS NO NOTABLE CORRELATION BETWEEN THE FEATURES]")
    st.markdown(
        "#### :white[THE DISTRIBUTION OF OUR FEATURES ARE NOT NORMALLY DISTRIBUTED LOOKS LIKE UNIFORM DISTRIBUTION]")

    st.markdown("## :white[DISTRIBUTION OF INCOME FEATURE]")
    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(df_s['Income'], kde=True)
    st.pyplot(fig)

    st.markdown("")
    st.markdown(
        "#### :white[THERE ARE SOME OUTLIERS PRESENT IN THE INCOME FEATURE]")
    st.markdown("")
    st.markdown(
        "#### :white[THESE ARE INCOME MORE THAN 3 STANDARD DEVIATION]")
    st.table(df_s[df_s.Income > df_s.Income.mean() + 3*df_s.Income.std()])

    df_1 = df_s[df_s.Income < df_s.Income.mean() + 3*df_s.Income.std()]

    df_2 = df_1.copy()

    df_2['Loan_Status'] = df_2['Loan_Status'].map(
        {'Non-Default': 0, 'Default': 1})
    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.heatmap(df_2.drop(
        ['Employment_Status', 'Location', 'Gender'], axis=1).corr(), cmap='coolwarm', annot=True)
    st.markdown(
        "#### :white[HEATMAP OF FEATURES GIVES MORE UNDERSTANDING ABOUT THE CORRELATION]")
    st.pyplot(fig)
    st.markdown(
        "#### :white[WE CANNOT FIT ANY LINEAR MODELS AS DISTRIBUTION AND CORRELATION ARE NOT GOOD]")
    st.markdown(
        "#### :white[MODELS THAT MIGHT NOT FIT WELL LIKE]")
    st.markdown(
        "#### :white[LOGISTIC REGRESSION]")
    st.markdown(
        "#### :white[SVC]")
    st.markdown(
        "#### :white[TREE BASED MODELS MIGHT WORKS]")
    st.markdown(
        "#### :white[DECISION TREE]")
    st.markdown(
        "#### :white[RANDOM FOREST]")
    st.markdown(
        "#### :white[XGBOOST]")
    st.markdown(
        "#### :white[DISTANCE BASED MODELS MIGHT WORKS]")
    st.markdown("#### :white[KNN-NEIGHBOURS]")

    st.markdown("#### :white[EXAMPLE]")
    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.regplot(y='Interest_Rate', x='Income', data=df_2)
    st.pyplot(fig)
    st.markdown(
        "#### :white[IN THE ABOVE PLOT WE CAN SEE HOW BAD LINEAR MODELS FITS]")
    rf = df_2[['Age', 'Income', 'Debt_to_Income_Ratio',
               'Existing_Loan_Balance', 'Credit_Score', 'Loan_Amount', 'Interest_Rate']]
    scaler = StandardScaler()
    scaler.fit(rf)
    scaled_data = scaler.transform(rf)
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    st.markdown(
        "#### :white[PRINCIPAL COMPONENT ANALYSIS]")
    fig = px.scatter(x_pca, x=x_pca[:, 0],
                     y=x_pca[:, 1], color=df_2['Loan_Status'])
    st.write(fig)
    fig = go.Figure(data=go.Scatter(
        x=x_pca[:, 0],
        y=x_pca[:, 1],
        mode='markers',
        marker=dict(
            size=16,
            color=df_2['Loan_Status'],  # set color equal to a variable
            colorscale='Viridis',  # one of plotly colorscales
            showscale=True
        )))

    st.write(fig)
    st.markdown(
        "#### :white[EVEN AFTER DIMENSION REDUCTUION THERE IS SO MUCH VARIANCE IN THE DATA AND ABSENCE OF AN DEFINITE PATTERN]")

    loan_status = df_2.groupby(['Employment_Status'])[
        'Loan_Status'].value_counts().reset_index()

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.barplot(data=loan_status, x='Employment_Status',
                     y='count', hue='Loan_Status')
    st.markdown(" ")
    st.markdown("#### :white[EMPLOYED VS UNEMPLOYED] ")
    st.pyplot(fig)

    loc = df_2.groupby(['Location'])[
        'Loan_Status'].value_counts().reset_index()

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.barplot(data=loc, x='Location', y='count', hue='Loan_Status')
    st.markdown(
        "#### :white[THE PROPORTION OF DEFAULTERS TENDS TO LOOK SIMILAR IN BOTH THE CATEGORIES] ")
    st.markdown("#### :white[LOCATION WISE LOAN DEFAULT] ")
    st.pyplot(fig)
    st.markdown(
        "#### :white[THERE IS NO MUCH DIFFERENCE IN PROPOTION OF DEFAULTERS IN EACH CATEGORIES] ")

    st.markdown("")
    st.markdown("#### :white[FEATURE WISE ANALYSIS WITH TARGET VARIABLE] ")
    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(data=df_2, x=df_2['Age'],
                      hue='Loan_Status', bins=50, kde=True)
    st.markdown("#### :white[AGE WISE DISTRIBUTION] ")
    st.pyplot(fig)
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO BE LOW IN THE AGE CATEGORY OF BELOW 20 AND BETWEEN THE AGE CATEGORY 30 AND 40.] ")
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO BE HIGHER BETWEEN 40 AND 50 AND ALSO ABOVE 63.]")
    st.markdown("")

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(
        data=df_2, x=df_2['Income'], hue='Loan_Status', bins=50, kde=True)
    st.markdown("#### :white[INCOME WISE DISTRIBUTION] ")
    st.pyplot(fig)
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO BE HIHGER IN THE INCOME CATEGORY 50000-60000 AND 80000-90000] ")
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO BE LOWER IN THE INCOME CATEGORY 20000-30000 AND 65000-75000] ")

    st.markdown("")

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(
        data=df_2, x=df_2['Credit_Score'], hue='Loan_Status', bins=50, kde=True)
    st.markdown("#### :white[CREDIT SCORE WISE DISTRIBUTION] ")
    st.pyplot(fig)
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO BE HIGHER BETWEEN CREDIT SCORE 500 AND 600 AND SLIGHTLLY NEAR 750 AND HIGHER ABOVE 810] ")
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO BE LOWER BETWEEN CREDIT SCORE 600 AND 700] ")

    st.markdown("")

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(
        data=df_2, x=df_2['Debt_to_Income_Ratio'], hue='Loan_Status', bins=50, kde=True)
    st.markdown("#### :white[DEBT TO INCOME RATIO DISTRIBUTION] ")
    st.pyplot(fig)
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO BE LOWER BETWEEN 0.2 TO 0.4 IN DEBT TO INCOME RATIO AND HIGHER AT 0.6 AND REDUCING TRENDS AFTER 0.8] ")

    st.markdown("")

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(
        data=df_2, x=df_2['Existing_Loan_Balance'], hue='Loan_Status', bins=50, kde=True)
    st.markdown("#### :white[EXISTING LOAN BALANCE DISTRIBUTION] ")
    st.pyplot(fig)
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO BE HIGHER IN EXISTING LOAN BALANCE BETWEEN 30000 TO 45000 AND REDUCING TREND AFTER 45000] ")

    st.markdown("")

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(
        data=df_2, x=df_2['Loan_Amount'], hue='Loan_Status', bins=50, kde=True)
    st.markdown("#### :white[LOAN AMOUNT DISTRIBUTION] ")
    st.pyplot(fig)
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO BE HIGHER FOR SMALLER VALUE LOAN BETWEEN 10000 TO 30000 AND REDUCING TREND FOR HIGHER VALUES] ")

    st.markdown("")

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(
        data=df_2, x=df_2['Loan_Duration_Months'], hue='Loan_Status', bins=50, kde=True)
    st.markdown("#### :white[LOAN DURATION MONTH WISE DISTRIBUTION] ")
    st.pyplot(fig)

    st.markdown("")

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(
        data=df_2, x=df_2['Interest_Rate'], hue='Loan_Status', bins=50, kde=True)
    st.markdown("#### :white[INTEREST RATE DISTRIBUTION] ")
    st.pyplot(fig)
    st.markdown(
        "#### :white[DEFAULTERS TENDS TO HIGHER BETWEEN INTEREST RATE OF 10 TO 15 PERCENT AND DECREASING TRENDS ABOVE 17.5 PERCENT] ")

    st.markdown("")
    df_2.drop('Gender', axis=1, inplace=True)
    df_3 = pd.get_dummies(
        columns=['Employment_Status', 'Location'], drop_first=True, dtype=int, data=df_2)
    X = df_3.drop('Loan_Status', axis=1)
    y = df_3['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=50)
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    feature_names = X_train.columns
    feature_imports = clf.feature_importances_
    most_imp_features = pd.DataFrame([f for f in zip(feature_names, feature_imports)], columns=[
                                     "Feature", "Importance"]).nlargest(11, "Importance")
    most_imp_features.sort_values(by="Importance", inplace=True)

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.barplot(data=most_imp_features, x=list(
        most_imp_features.Importance), y=list(most_imp_features.Feature), orient='h')
    st.markdown(
        "#### :white[FEATURE IMPORTANCE BY DECISION TREE CRITERION=GINI] ")
    st.pyplot(fig)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=50)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, y)

    feature_names = X_train.columns
    feature_imports = clf.feature_importances_
    most_imp_features = pd.DataFrame([f for f in zip(feature_names, feature_imports)], columns=[
                                     "Feature", "Importance"]).nlargest(11, "Importance")
    most_imp_features.sort_values(by="Importance", inplace=True)

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.barplot(data=most_imp_features, x=list(
        most_imp_features.Importance), y=list(most_imp_features.Feature), orient='h')
    st.markdown(
        "#### :white[FEATURE IMPORTANCE BY DECISION TREE CRITERION=ENTROPY] ")
    st.pyplot(fig)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    feature_names = X_train.columns
    feature_imports = clf.feature_importances_
    most_imp_features = pd.DataFrame([f for f in zip(feature_names, feature_imports)], columns=[
                                     "Feature", "Importance"]).nlargest(11, "Importance")
    most_imp_features.sort_values(by="Importance", inplace=True)
    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.barplot(data=most_imp_features, x=list(
        most_imp_features.Importance), y=list(most_imp_features.Feature), orient='h')
    st.markdown(
        "#### :white[FEATURE IMPORTANCE BY RANDOM FOREST CLASSIFIER CRITERION=GINI] ")
    st.pyplot(fig)

    lf = RandomForestClassifier(criterion='entropy')
    clf.fit(X_train, y_train)
    feature_names = X_train.columns
    feature_imports = clf.feature_importances_
    most_imp_features = pd.DataFrame([f for f in zip(feature_names, feature_imports)], columns=[
                                     "Feature", "Importance"]).nlargest(11, "Importance")
    most_imp_features.sort_values(by="Importance", inplace=True)
    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.barplot(data=most_imp_features, x=list(
        most_imp_features.Importance), y=list(most_imp_features.Feature), orient='h')
    st.markdown(
        "#### :white[FEATURE IMPORTANCE BY RANDOM FOREST CLASSIFIER CRITERION=ENTROPY] ")
    st.pyplot(fig)


if selected == 'PREDICTION':
    df_s = pd.read_csv('loan_default_prediction_project(1).csv')
    df_1 = df_s[df_s.Income < df_s.Income.mean() + 3*df_s.Income.std()]
    df_2 = df_1.copy()

    df_2['Loan_Status'] = df_2['Loan_Status'].map(
        {'Non-Default': 0, 'Default': 1})
    df_2.drop('Gender', axis=1, inplace=True)
    df_3 = pd.get_dummies(
        columns=['Employment_Status', 'Location'], drop_first=True, dtype=int, data=df_2)
    X = df_3.drop('Loan_Status', axis=1)
    y = df_3['Loan_Status']
    st.markdown("#### :white[PROCESSED SAMPLE DATA]")
    st.table(X.head())

    smote = SMOTE(sampling_strategy='minority')
    X_sm, y_sm = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.049999, max_depth=20, subsample=0.7)
    xgb.fit(X_train, y_train)
    st.markdown("")
    st.markdown("")
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input(label="Age", label_visibility="visible")
    with col2:
        Income = st.text_input(
            label="Income", label_visibility="visible")
    with col3:
        Credit_Score = st.text_input(
            label="Credit_Score", label_visibility="visible")

    with col1:
        Debt_to_Income_Ratio = st.text_input(
            label="Debt_to_Income_Ratio", label_visibility="visible")
    with col2:
        Existing_Loan_Balance = st.text_input(
            label="Existing_Loan_Balance", label_visibility="visible")
    with col3:
        Loan_Amount = st.text_input(
            label="Loan_Amount", label_visibility="visible")

    with col1:
        Interest_Rate = st.text_input(
            label="Interest_Rate", label_visibility="visible")
    with col2:
        Loan_Duration_Months = st.text_input(
            label="Loan_Duration_Months", label_visibility="visible")
    with col3:
        Employment_Status_Unemployed = st.text_input(
            label="Employment_Status_Unemployed", label_visibility="visible")

    with col1:
        Location_Suburban = st.text_input(
            label="Location_Suburban", label_visibility="visible")
    with col2:
        Location_Urban = st.text_input(
            label="Location_Urban", label_visibility="visible")

    if st.button("PREDICT"):

        pred = xgb.predict_proba([[int(Age), float(Income), float(Credit_Score), float(Debt_to_Income_Ratio), float(Existing_Loan_Balance), float(Loan_Amount),
                                   float(Interest_Rate), int(Loan_Duration_Months), int(Employment_Status_Unemployed), int(Location_Suburban), int(Location_Urban)]])
        s = pd.DataFrame(
            pred, index=["1"], columns=["NON_DEFAULT_PROBABILITY", "DEFAULT_PROBABILITY"])
        st.table(s)
