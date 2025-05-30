import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Read the features and classification data
features_df = pd.read_csv('features.csv')
class_df = pd.read_excel('train/classif.xlsx')

# Remove the entry with ID 154 as it was removed from features
class_df = class_df[class_df['ID'] != 154]

# Merge features with classification data
merged_df = pd.merge(features_df, class_df, left_on='img_name', right_on='ID', how='inner')

# 1. Basic Feature Analysis
def plot_feature_distributions(df):
    # Select numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create subplots for each feature
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        sns.histplot(data=df, x=col, hue='bug type', ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

# 2. Correlation Analysis
def plot_correlation_matrix(df):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

# 3. PCA Visualization
def plot_pca_analysis(df):
    # Select features for PCA
    feature_cols = ['r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std', 
                    'ratio', 'convex_ratio', 'exentricity']
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Create PCA plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=df['bug type'].astype('category').cat.codes,
                         cmap='viridis')
    plt.title('PCA of Features')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, label='Bug Type')
    plt.savefig('pca_analysis.png')
    plt.close()

# 4. Interactive 3D Plot
def create_3d_plot(df):
    # Select features for 3D plot
    feature_cols = ['r_mean', 'g_mean', 'b_mean']
    
    fig = px.scatter_3d(df, 
                        x='r_mean', 
                        y='g_mean', 
                        z='b_mean',
                        color='bug type',
                        title='3D Color Space Distribution')
    
    fig.write_html('color_space_3d.html')

# 5. Feature Importance Analysis
def plot_feature_importance(df):
    # Calculate mean values by bug type
    feature_cols = ['r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std', 
                    'ratio', 'convex_ratio', 'exentricity']
    
    mean_by_type = df.groupby('bug type')[feature_cols].mean()
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    mean_by_type.plot(kind='bar')
    plt.title('Mean Feature Values by Bug Type')
    plt.xlabel('Bug Type')
    plt.ylabel('Feature Value')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Execute all visualizations
plot_feature_distributions(merged_df)
plot_correlation_matrix(merged_df)
plot_pca_analysis(merged_df)
create_3d_plot(merged_df)
plot_feature_importance(merged_df)

print("All visualizations have been generated. Check the following files:")
print("- feature_distributions.png")
print("- correlation_matrix.png")
print("- pca_analysis.png")
print("- color_space_3d.html")
print("- feature_importance.png") 