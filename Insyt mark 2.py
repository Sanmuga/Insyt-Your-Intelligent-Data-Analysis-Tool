import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to display confusion matrix heatmap
def display_confusion_matrix(confusion_mat):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(confusion_mat, cmap="Blues", interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(j, i, str(confusion_mat[i, j]), ha="center", va="center")
    st.pyplot(fig)

# Function to display regression predictions
def display_regression_predictions(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='black')
    ax.set_title("Regression Prediction")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    st.pyplot(fig)

# Function to display scatter plot
def display_scatter_plot(data, x_col, y_col):
    fig, ax = plt.subplots()
    ax.scatter(data[x_col], data[y_col], color='blue')
    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

# Function to display bar chart
def display_bar_chart(data, x_col, y_col):
    fig, ax = plt.subplots()
    ax.bar(data[x_col], data[y_col], color='green')
    ax.set_title(f'Bar Chart: {x_col} vs {y_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

# Function to display histogram
def display_histogram(data, col):
    fig, ax = plt.subplots()
    ax.hist(data[col], bins=20, color='purple')
    ax.set_title(f'Histogram: {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Function to display line plot
def display_line_plot(data, x_col, y_col):
    fig, ax = plt.subplots()
    ax.plot(data[x_col], data[y_col], color='red')
    ax.set_title(f'Line Plot: {x_col} vs {y_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Insyt: Intelligent Data Analysis Tool")

    st.markdown("""
    Welcome to **Insyt**, your intelligent data analysis tool for classification and regression tasks.
    Upload a CSV file, select target variables, and analyze your data effortlessly!
    """)

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the dataset into a pandas DataFrame
        dataset = pd.read_csv(uploaded_file)

        # Display the first few rows of the dataset
        st.write("Dataset:")
        st.write(dataset.head())

        # Get all the columns and store them in a variable
        all_columns = dataset.columns

        # Display the list of columns
        st.write("\nAll Columns:", all_columns)

        # Ask the user for the target variable
        target_column = st.selectbox("Select the target variable:", all_columns)

        # Ask the user to select columns for analysis
        selected_columns = st.multiselect("Select columns for analysis:", all_columns)

        if not selected_columns:
            st.warning("Please select at least one column for analysis.")
            return

        # Ask the user to select columns for plotting
        plot_columns = st.multiselect("Select columns for plotting:", all_columns)

        if plot_columns:
            plot_type = st.selectbox("Select plot type:", ['Scatter Plot', 'Bar Chart', 'Histogram', 'Line Plot'])

            if plot_type == 'Scatter Plot' and len(plot_columns) >= 2:
                display_scatter_plot(dataset, plot_columns[0], plot_columns[1])
            elif plot_type == 'Bar Chart' and len(plot_columns) >= 2:
                display_bar_chart(dataset, plot_columns[0], plot_columns[1])
            elif plot_type == 'Histogram' and len(plot_columns) >= 1:
                display_histogram(dataset, plot_columns[0])
            elif plot_type == 'Line Plot' and len(plot_columns) >= 2:
                display_line_plot(dataset, plot_columns[0], plot_columns[1])
            else:
                st.warning("Please select appropriate columns for the selected plot type.")

        # Separate features (X) and target variable (y)
        X = dataset[selected_columns]
        y = dataset[target_column]

        # Perform one-hot encoding for categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)

        # PCA integration
        apply_pca = st.checkbox("Apply PCA")
        if apply_pca:
            n_components = st.slider("Select number of principal components:", 1, min(X_encoded.shape[1], 10))
            pca = PCA(n_components=n_components)
            X_encoded = pca.fit_transform(X_encoded)
            st.write(f"PCA applied with {n_components} components")

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Ask the user for the classification or prediction task
        task = st.radio("Choose task:", ['Classification', 'Regression'])

        if task == 'Classification':
            # Train a decision tree classifier
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            confusion_mat = confusion_matrix(y_test, y_pred)

            # Display results
            st.write("\nAccuracy:", accuracy)
            st.write("\nClassification Report:\n", report)
            st.write("\nConfusion Matrix:\n", confusion_mat)

            # Visualize the confusion matrix
            display_confusion_matrix(confusion_mat)

        elif task == 'Regression':
            # Train a linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Display results
            st.write("\nMean Squared Error:", mse)
            st.write("\nR-squared Score:", r2)

            # Visualize the predictions
            display_regression_predictions(y_test, y_pred)

        else:
            st.warning("Invalid task. Please choose 'Classification' or 'Regression'.")

if __name__ == "__main__":
    main()