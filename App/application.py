import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense

# Set up the page configuration
icon = Image.open("chef.jpg")
st.set_page_config(layout='centered', page_title='AI-Powered Recipe Recommender', page_icon=icon)

# Upload and display the project logo
st.image(Image.open("project_logo.JPG"), use_column_width=True)

# Load the saved models and components
with open('recipe_recommendation_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('pca_model.pkl', 'rb') as file:
    pca = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# Use the cached function to load the data
df = load_data('all_recipes_final_df_v2.csv')

# Update the columns to reflect grams with daily percentage
df['Carbohydrates g(Daily %)'] = df.apply(lambda x: f"{x['carbohydrates_g']}g ({x['carbohydrates_g_dv_perc']}%)", axis=1)
df['Sugars g(Daily %)'] = df.apply(lambda x: f"{x['sugars_g']}g ({x['sugars_g_dv_perc']}%)", axis=1)
df['Fat g(Daily %)'] = df.apply(lambda x: f"{x['fat_g']}g ({x['fat_g_dv_perc']}%)", axis=1)
df['Protein g(Daily %)'] = df.apply(lambda x: f"{x['protein_g']}g ({x['protein_g_dv_perc']}%)", axis=1)

# Transform the combined features using the loaded TF-IDF vectorizer and PCA model
tfidf_matrix = tfidf.transform(df['combined_features'])  # Use transform instead of fit_transform
tfidf_pca = pca.transform(tfidf_matrix.toarray())  # Use transform instead of fit_transform

# Rename the columns to user-friendly names
friendly_names = {
    'name': 'Recipe Name',
    'category': 'Category',
    'calories': 'Calories (kcal)',
    'servings': 'Servings',
    'Carbohydrates g(Daily %)': 'Carbohydrates g(Daily %)',
    'Sugars g(Daily %)': 'Sugars g(Daily %)',
    'Fat g(Daily %)': 'Fat g(Daily %)',
    'Protein g(Daily %)': 'Protein g(Daily %)',
    'cook': 'Cook Time (minutes)',
    'rating': 'Rating',
    'rating_count': 'Rating Count',
    'diet_type': 'Diet Type',
    'ingredients': 'Ingredients',
    'directions': 'Directions'
}

# Function to get similar recipes
def get_similar_recipes(recipe_name, top_n=5, diversify=False, diversity_factor=0.1):
    target_index = df[df['name'] == recipe_name].index[0]
    target_features = tfidf.transform([df['combined_features'].iloc[target_index]])
    target_features_pca = pca.transform(target_features.toarray())
    target_cluster = model.predict(target_features_pca).argmax()
    cluster_indices = df[df['cluster'] == target_cluster].index
    similarities = cosine_similarity(target_features_pca, tfidf_pca[cluster_indices]).flatten()
    weighted_similarities = similarities * df.loc[cluster_indices, 'rating']
    
    if diversify:
        diversified_scores = weighted_similarities * (1 - diversity_factor * np.arange(len(weighted_similarities)))
        similar_indices = cluster_indices[np.argsort(diversified_scores)[-top_n:][::-1]]
    else:
        similar_indices = cluster_indices[np.argsort(weighted_similarities)[-top_n:][::-1]]
    
    # Retrieve similar recipes and sort them by rating_count and rating
    similar_recipes = df.iloc[similar_indices]
    similar_recipes_sorted = similar_recipes.sort_values(by=['rating_count', 'rating'], ascending=False)
    
    # Select only the desired columns
    selected_columns = ['name', 'category', 'ingredients', 'directions','rating', 'rating_count', 'diet_type','calories', 'servings', 'Carbohydrates g(Daily %)', 'Sugars g(Daily %)', 'Fat g(Daily %)', 'Protein g(Daily %)', 'cook']
    selected_recipes = similar_recipes_sorted[selected_columns].head(top_n)
    
    return selected_recipes.rename(columns=friendly_names)

# Function to filter recipes by servings
def filter_by_servings(servings):
    if servings == "one":
        return df[df['servings'] == 1]
    elif servings == "two":
        return df[df['servings'] == 2]
    elif servings == "crowd":
        return df[df['servings'] >= 5]
    else:
        return pd.DataFrame()

# Function to filter recipes by name
def filter_by_recipe_name(name):
    return df[df['name'].str.contains(name, case=False, na=False)]

# Function to autocomplete suggestions
def autocomplete_suggestions(user_input, df, max_suggestions=5):
    # Filter recipe names that contain the user input
    filtered_df = df[df['name'].str.contains(user_input, case=False, na=False)]
    
    # Sort by rating_count to prioritize popular recipes
    sorted_df = filtered_df.sort_values(by='rating_count', ascending=False)
    
    # Return the top `max_suggestions` recipe names
    return sorted_df['name'].head(max_suggestions).tolist()

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Introduction section
st.markdown(
    """
    ## Introduction
    
    Are you in search of the perfect recipe? Look no further! Our Recipe Recommendation App is designed to help you discover delicious recipes tailored to your preferences. Whether you're searching for something specific or exploring new dishes, this app offers a variety of features to enhance your culinary journey.
    
    #### Search Options:
    
    **1) Personalized Recommendations:**
    - Simply enter the name of a recipe, and the app will suggest similar recipes tailored to your preferences.
    - Autocomplete suggestions guide you to the exact recipe name you're looking for.
    - Our recommender system is based on Rating, Category, Diet Type, and Ingredients.
    
    **2) Popular Searches:**
    - Quickly access popular and trending recipes.
    - Search by common keywords like "Chicken," "Pancakes," or "Lasagna."
    - Filter by serving size for tailored results.
    
    **3) Custom Search:**
    - Select recipes based on categories like "Main Dish," "Desserts," or "World Cuisine."
    - Filter by diet type, such as "Low Carb" or "High Protein."
    - Choose recipes based on serving size or cooking time, perfect for specific meal planning needs.
    """
)

st.write('---')

st.markdown(
    """
    ## Find Your Perfect Recipe
    """
)

option = st.selectbox(
    'How would you like to search for recipes?',
    ('Personalized Recommendations', 'Popular Searches', 'Custom Search')
)

if option == 'Personalized Recommendations':
    st.write(
"""
    ### **Get Recommendations**
""") 
    
    # Input field with suggestions
    user_input = st.text_input('Enter a Recipe Name')
    
    suggestions = []
    if user_input:
        suggestions = autocomplete_suggestions(user_input, df)

    selected_recipe = None
    if suggestions:
        st.write("Did you mean:")
        for suggestion in suggestions:
            if st.button(suggestion):
                selected_recipe = suggestion
                break  # Exit the loop once a selection is made

    if selected_recipe:
        similar_recipes = get_similar_recipes(selected_recipe, top_n=10, diversify=False)
        st.write(f"Top 10 recommendations for '{selected_recipe}':")
        
        # Display the recipes in a blog format
        for index, row in similar_recipes.iterrows():
            st.markdown(f"### {row['Recipe Name']}")
            st.markdown(f"**Category:** {row['Category']}")
            st.markdown(f"**Diet Type:** {row['Diet Type']}")
            st.markdown(f"**Rating:** {row['Rating']} ({row['Rating Count']} ratings)")
            st.markdown(f"**Servings:** {row['Servings']}")
            st.markdown(f"**Calories:** {row['Calories (kcal)']}")
            st.markdown(f"**Cook Time:** {row['Cook Time (minutes)']} minutes")
            st.markdown(f"**Ingredients:** {row['Ingredients']}")
            st.markdown(f"**Directions:** {row['Directions']}")
            st.write("---")

        st.write("#### Learn More")
        st.markdown("[![](https://via.placeholder.com/400x200.png?text=Explore+More)](https://www.example.com)")
    
elif option == 'Popular Searches':
    st.write(
"""
    ### **Popular Searches**
""")
    
    # Display popular recipes (Top 5 based on rating)
    popular_recipes = df.sort_values(by='rating_count', ascending=False).head(5)
    
    # Display the recipes in a blog format
    for index, row in popular_recipes.iterrows():
        st.markdown(f"### {row['name']}")
        st.markdown(f"**Category:** {row['category']}")
        st.markdown(f"**Diet Type:** {row['diet_type']}")
        st.markdown(f"**Rating:** {row['rating']} ({row['rating_count']} ratings)")
        st.markdown(f"**Servings:** {row['servings']}")
        st.markdown(f"**Calories:** {row['calories']} kcal")
        st.markdown(f"**Cook Time:** {row['cook']} minutes")
        st.markdown(f"**Ingredients:** {row['ingredients']}")
        st.markdown(f"**Directions:** {row['directions']}")
        st.write("---")

elif option == 'Custom Search':
    st.write(
"""
    ### **Custom Search**
""")
    
    # Filter options
    servings_option = st.selectbox('Choose serving size:', ['All', 'one', 'two', 'crowd'])
    
    if servings_option != 'All':
        filtered_recipes = filter_by_servings(servings_option)
        st.write(f"Found {len(filtered_recipes)} recipes for serving size '{servings_option}'.")
        
        # Display filtered recipes
        for index, row in filtered_recipes.iterrows():
            st.markdown(f"### {row['name']}")
            st.markdown(f"**Category:** {row['category']}")
            st.markdown(f"**Diet Type:** {row['diet_type']}")
            st.markdown(f"**Rating:** {row['rating']} ({row['rating_count']} ratings)")
            st.markdown(f"**Servings:** {row['servings']}")
            st.markdown(f"**Calories:** {row['calories']} kcal")
            st.markdown(f"**Cook Time:** {row['cook']} minutes")
            st.markdown(f"**Ingredients:** {row['ingredients']}")
            st.markdown(f"**Directions:** {row['directions']}")
            st.write("---")
    
    # Displaying the total number of recipes
    total_recipes = len(df)
    st.write(f"Total recipes available: {total_recipes}")

# Footer with information about the app
st.markdown(
    """
    ---
    ### About This App
    This app was developed as a part of a project to help users discover and recommend recipes. 
    It utilizes machine learning techniques to provide personalized recipe recommendations based on user input.
    
    ### Feedback
    We welcome your feedback! Please share your thoughts to help us improve the app.
    
    ### Contact
    For any inquiries, please reach out to [Your Email Here].
    """
)

