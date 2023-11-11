import holoviews as hv
import matplotlib
import panel as pn
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.switch_backend('agg')
matplotlib.use('agg')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import plotly.express as px

# ... (rest of your code
import hvplot.pandas
import shap
import hvplot
# Load the dataset
data = pd.read_csv("C:/Users/21656/Desktop/Projet_Dataviz/awesome-panel/Projet_Data_Viz/Data/dataset.csv")
data = data.dropna()
# Your feature engineering code here...
genre_mapping = {
    'Rock and Subgenres': ['alt-rock', 'alternative', 'grunge', 'punk-rock', 'club','emo','drum-and-bass','hard-rock','rock', 'rock-n-roll','psych-rock','industrial','rockabilly','punk'],
    'Dance and Upbeat Genres': ['dance','dancehall','electro','hardcore','hardstyle','chicago-house','progressive-house','deep-house','disco','edm','electronic','house','techno','trance'],
    'Chill and Experimental Genres': ['acoustic','ambient','detroit-techno','idm','breakbeat','dubstep','minimal-techno'],
    'Pop and Subgenres': ['pop-film', 'pop', 'power-pop','indie-pop','synth-pop'],
    'Hip-Hop and R&B': ['hip-hop', 'r-n-b','trip-hop'],
    'Metal and Subgenres': ['black-metal', 'death-metal', 'heavy-metal', 'metal', 'metalcore','grindcore','goth'],
    'Folk and Country': ['bluegrass', 'country', 'folk', 'honky-tonk'],
    'Latin and Subgenres': ['brazil', 'latin', 'latino', 'salsa', 'samba', 'sertanejo','tango','spanish','forro','pagode','mpb'],
    'World and Traditional': ['afrobeat','british','swedish','french', 'indian', 'iranian', 'malay', 'mandopop', 'turkish', 'german','world-music','reggae', 'reggaeton','dub'],
    'J-Pop and Asian Pop': ['j-dance', 'j-idol', 'j-pop', 'j-rock', 'k-pop','cantopop'],
    'Classical and Opera': ['classical', 'opera','piano'],
    'Soulful Groove': ['blues', 'jazz','funk', 'groove', 'soul'],
    'Diverse and Uplifting Genres':['chill','anime','kids','disney','guitar','children','comedy','gospel','party','romance','indie','show-tunes','ska'],
    'Reflective and Relaxing Genres':['garage','sad','new-age','happy','study','sleep','songwriter','singer-songwriter']
}

data['top_level_genre'] = data['track_genre'].apply(lambda x: next((k for k, v in genre_mapping.items() if x in v), 'Other'))



# Encode categorical columns
le = LabelEncoder()
data['artists'] = le.fit_transform(data['artists'])
data['album_name'] = le.fit_transform(data['album_name'])
data['track_name'] = le.fit_transform(data['track_name'])
# 2. Create Interaction Features (e.g., 'energy_loudness_interaction')
#data['energy_loudness_interaction'] = data['energy'] * data['loudness']
#genre_mean_energy = data.groupby('track_genre')['energy'].mean()

#data['mean_energy_per_genre'] = data['track_genre'].map(genre_mean_energy)
data['Spch-Acous. Int.'] = data['speechiness'] * data['acousticness']

# 3. Aggregated Features (e.g., mean popularity per genre)
aggregated_features = data.groupby('track_genre')['popularity'].mean()
data['Popul. Std Dev'] = abs(data['track_genre'].map(aggregated_features)-data['popularity'])

#data['top_level_genre'] = le.fit_transform(data['top_level_genre'])
aggregated_features = data.groupby('track_genre')['acousticness'].mean()
data['Acous. Std Dev'] = abs(data['track_genre'].map(aggregated_features)-data['acousticness'])




data.to_csv('cleaned_data.csv', index=False)
#def create_dash(data):
# Create a StandardScaler instance
def create_dash(data,top_level_genres):
    scaler = StandardScaler()
    # Define a variable to track if the model is fitted
    global model_fitted
    model_fitted=False
    # Train a Random Forest classifier for top-level genre classification
    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    svm_classifier = SVC(kernel='linear', random_state=42)  # You can configure the SVM hyperparameters here

    # Create a Panel dashboard
    pn.extension()
    pn.extension('plotly')


    # Define a function to update the dataset based on the selected top-level genre
    def update_dataset(event):
        global selected_genre
        selected_genre = event.new
        global X_train, X_test, y_train, y_test, X_genre

        # Filter the data for the selected top-level genre
        filtered_data = data[data['top_level_genre'] == selected_genre]

        # Split the data into features and labels for the selected genre
        X_genre = filtered_data.drop(['Unnamed: 0', 'track_id', 'track_genre', 'top_level_genre'], axis=1)
        X_genre_norm = scaler.fit_transform(X_genre)
        y_genre = filtered_data['track_genre']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_genre_norm, y_genre, test_size=0.2, random_state=42)

        # Update the model and status
        model_fitted = False

    # Create a widget for selecting the top-level genre
    genre_selector = pn.widgets.Select(name="Select Top-Level Genre",value='' ,options=list(top_level_genres))
    genre_selector.param.watch(update_dataset, 'value')


    # Define a function to train the model
    def train_model(event):
        global model,feature_importances
        model_fitted = True
        if selected_model.value == "Random Forest":
            progress_bar.active = True
            rf_classifier.fit(X_train, y_train)
            feature_importances = rf_classifier.feature_importances_
            model = rf_classifier
            progress_bar.active = False
            test_button.disabled = False
            test.value="oiojdeoidjoe"
        elif selected_model.value == "SVM":
            progress_bar.active = True
            svm_classifier.fit(X_train, y_train)
            feature_importances = svm_classifier.coef_[0]
            model = svm_classifier
            progress_bar.active = False
            test_button.disabled = False
        
    
    
    # Define a function to test the model
    def test_model(event):
        
        #if not model_fitted:
            
           # return
        model_fitted=False
        progress_bar_test.active = True
        y_top_level_pred = model.predict(X_test)
        cm = pd.DataFrame(confusion_matrix(y_test, y_top_level_pred),columns=genre_mapping[selected_genre],
                          index=genre_mapping[selected_genre])
        report = classification_report(y_test, y_top_level_pred)

        accuracy_text.value = f"Accuracy: {accuracy_score(y_test, y_top_level_pred):.2f}"

        # Display the heatmap in a Panel pane
        cf=pd.DataFrame(cm)
        confusion_matrix_pane.object=cf
        classification_report_pane.object = report
        model_param.object = model.get_params()
        progress_bar_test.active = False
        

    n_estimators_slider = pn.widgets.IntSlider(name="Number of Estimators", start=10, end=200, step=10, value=100)
    max_depth_slider = pn.widgets.IntSlider(name="Max Depth", start=1, end=30, value=10)
    min_samples_split_slider = pn.widgets.FloatSlider(name="Min Samples Split", start=0.01, end=0.5, value=0.2)
    min_samples_leaf_slider = pn.widgets.FloatSlider(name="Min Samples Leaf", start=0.01, end=0.5, value=0.1)
    random_state_slider = pn.widgets.IntSlider(name="Random State", start=0, end=100, value=42)


    # Define a function to update the model with adjusted hyperparameters
    def update_hyperparameters(event):
        rf_classifier.set_params(
            n_estimators=n_estimators_slider.value,
            max_depth=max_depth_slider.value,
            min_samples_split=min_samples_split_slider.value,
            min_samples_leaf=min_samples_leaf_slider.value,
            random_state=random_state_slider.value

        )
        

    # Update the model with new hyperparameters when the user changes them
    n_estimators_slider.param.watch(update_hyperparameters, 'value')
    max_depth_slider.param.watch(update_hyperparameters, 'value')
    min_samples_split_slider.param.watch(update_hyperparameters, 'value')
    min_samples_leaf_slider.param.watch(update_hyperparameters, 'value')
    random_state_slider.param.watch(update_hyperparameters, 'value')

    def visualize_feature_importance(event):
        # Calculate feature importances (assuming 'model' and 'X_genre' are defined)
        feature_names = X_genre.columns
        top_features = pd.Series(feature_importances, index=feature_names).nlargest(10)
        top_features = top_features.reset_index()
        top_features.columns = ['Feature', 'Importance']

        # Create an hvPlot bar plot
        feature_importance_plot = top_features.hvplot.bar(x='Feature', y='Importance', title='Feature Importance', width=1000,xrotation=45)

        # Update the feature_importance_pane with the new plot
        feature_importance_pane.object = feature_importance_plot


    # Create Panel widgets
    train_button = pn.widgets.Button(name="Train Model", button_type="primary")
    train_button.on_click(train_model)

    test_button = pn.widgets.Button(name="Test Model", button_type="primary")
    test_button.on_click(test_model)

    accuracy_text = pn.widgets.TextInput(value=f"Accuracy: {0.0}", disabled=True)
    test = pn.widgets.TextInput(value=f"Accuracy: {0.0}", disabled=True)

    confusion_matrix_pane = pn.pane.DataFrame(None)


    classification_report_pane = pn.pane.JSON(None)
    progress_bar = pn.widgets.Progress(name="Training Progress", active=False,width=300, height=50)
    progress_bar_test = pn.widgets.Progress(name="Testing Progress", active=False,width=300, height=50)

    model_param=pn.pane.JSON(None)

    Empty=pd.DataFrame([[0]])
    # Create a Panel widget for displaying the feature importance plot
    feature_importance_pane = pn.pane.HoloViews(None)
    # Create a button to trigger feature importance visualization
    feature_importance_button = pn.widgets.Button(name="Visualize Feature Importance", button_type="success")
    feature_importance_button.on_click(visualize_feature_importance)  # Call the visualize_feature_importance function when clicked


    selected_model = pn.widgets.Select(name="Select Model", value='Random Forest', options=['Random Forest', 'SVM'])

    # Explanation of Feature Importance
    feature_importance_text = """
        The Feature Importance plot provides insights into the contribution of each feature
        to the model's decision-making process. Features with higher importance values have a
        greater impact on the model's predictions. In this context:

        - 'Spch-Acous. Int.': Captures the interaction between speechiness and acousticness.
        - 'Popul. Std Dev': Measures the popularity deviation from the genre mean.
        - 'Acous. Std Dev': Measures the acousticness deviation from the genre mean.

        Understanding feature importance is crucial for interpreting the model's behavior and
        gaining insights into the key factors influencing the classification of music genres.
        """




        # Create a sunburst chart for genre mapping visualization
    def visualize_genre_mapping(event):
            genre_hierarchy = {
                'Top-Level Genre': list(genre_mapping.keys()),
                'Subgenres': [', '.join(subgenres) for subgenres in genre_mapping.values()]
            }

            fig = px.sunburst(
                genre_hierarchy,
                path=['Top-Level Genre', 'Subgenres'],
                title='Genre Mapping Visualization',
                height=600
            )

            # Display the sunburst chart in a Panel pane
            genre_mapping_pane.object = fig

# ... (rest of your code)

# Create Panel widgets  
    genre_mapping_button = pn.widgets.Button(name="Visualize Genre Mapping", button_type="primary")
    genre_mapping_button.on_click(visualize_genre_mapping)

    genre_mapping_pane = pn.pane.Plotly(None)
    mapping_explanation_text = """
    Due to the substantial number of track genres (114) and the limited 
    correlation between track genre and other variables, we opted to 
    categorize the dataset into top-level genres for more effective analysis.

    The extensive diversity in genres made it challenging to establish 
    significant correlations. To address this, we created a mapping of each 
    track to a top-level genre based on genre similarities.
    
    This approach allows us to perform more focused analyses on subsets of 
    the data, making it more manageable and facilitating better model
    performance on each sub-genre dataset.
    """
    # Define your Panel dashboard layout
    dashboard = pn.Row(
    pn.Column(
           """
            # Exploring Spotify Track Genres

            Welcome to the Spotify Track Genres Explorer! This interactive 
            dashboard is designed to answer the business question: How can
            we explain and predict the genre of a track? Use the dropdowns 
            and buttons to train and test machine learning models, visualize
            feature importance, and explore the results of the analysis.

            """,mapping_explanation_text,"<h1 style='color: black;'>Top-Level Genre Classifier</h1>", 
        selected_model,
        genre_selector,
        train_button,
        progress_bar,
        test_button,
        progress_bar_test,
        accuracy_text,
        feature_importance_button,genre_mapping_button
    ),
    pn.Column(
        pn.Row(
            pn.Column("<h2 style='color: green;'>Model Parameters</h2>",  # HTML styling for sub-header
                model_param),
            pn.Column("<h2 style='color: green;'>Classification Report</h2>",  # HTML styling for sub-header
                classification_report_pane)
        ),
        pn.Row(
            pn.Column("<h2 style='color: green;'>Confusion Matrix</h2>",  # HTML styling for sub-header
                confusion_matrix_pane)
        ),
        pn.Row(
            pn.Column("<h2 style='color: purple;'>Feature Importance</h2>",  # HTML styling for sub-header
                feature_importance_text,feature_importance_pane,genre_mapping_pane)
        )
    )
    )

    return dashboard



