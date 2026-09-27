"""
Green Skilling Sustainable Goals Analytics Dashboard
Streamlit App for Automated Data Analysis and Visualization

This app automatically analyzes uploaded CSV files and generates comprehensive
visualizations with AI-powered insights for sustainable development goals.

Run with: streamlit run green_skilling_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Green Skilling Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f8f0;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1b5e20;
    }
    h2 {
        color: #2e7d32;
    }
    .insight-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class DataAnalyzer:
    """Comprehensive data analysis class"""
    
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def get_summary_statistics(self):
        """Generate summary statistics"""
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_cols),
            'categorical_columns': len(self.categorical_cols),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        return summary
    
    def detect_column_types(self):
        """Automatically detect column purposes"""
        sustainability_keywords = ['green', 'carbon', 'renewable', 'energy', 'emission', 
                                  'sustainable', 'environment', 'eco', 'solar', 'wind']
        skill_keywords = ['skill', 'training', 'education', 'course', 'certification', 
                         'learning', 'competency', 'qualification']
        
        detected = {
            'sustainability_cols': [],
            'skill_cols': [],
            'temporal_cols': [],
            'geographic_cols': []
        }
        
        for col in self.df.columns:
            col_lower = col.lower()
            
            if any(keyword in col_lower for keyword in sustainability_keywords):
                detected['sustainability_cols'].append(col)
            
            if any(keyword in col_lower for keyword in skill_keywords):
                detected['skill_cols'].append(col)
            
            if 'date' in col_lower or 'time' in col_lower or 'year' in col_lower:
                detected['temporal_cols'].append(col)
            
            if 'country' in col_lower or 'region' in col_lower or 'location' in col_lower:
                detected['geographic_cols'].append(col)
        
        return detected
    
    def generate_insights(self, plot_type, data_used):
        """Generate AI-powered insights for visualizations"""
        insights = []
        
        if plot_type == "distribution":
            col = data_used['column']
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            std_val = self.df[col].std()
            skewness = self.df[col].skew()
            
            insights.append(f"📊 **Distribution Analysis for {col}:**")
            insights.append(f"- Mean: {mean_val:.2f}, Median: {median_val:.2f}")
            insights.append(f"- Standard Deviation: {std_val:.2f}")
            
            if skewness > 0.5:
                insights.append(f"- ⚠️ Right-skewed distribution (skewness: {skewness:.2f}) - Most values are concentrated on the lower end")
            elif skewness < -0.5:
                insights.append(f"- ⚠️ Left-skewed distribution (skewness: {skewness:.2f}) - Most values are concentrated on the higher end")
            else:
                insights.append(f"- ✅ Approximately symmetric distribution (skewness: {skewness:.2f})")
        
        elif plot_type == "correlation":
            insights.append("🔗 **Correlation Analysis:**")
            insights.append("- Strong positive correlations (>0.7) indicate variables that increase together")
            insights.append("- Strong negative correlations (<-0.7) indicate inverse relationships")
            insights.append("- Values near 0 suggest weak or no linear relationship")
            
        elif plot_type == "time_series":
            col = data_used['column']
            trend = np.polyfit(range(len(self.df)), self.df[col].fillna(0), 1)[0]
            
            insights.append(f"📈 **Temporal Analysis for {col}:**")
            if trend > 0:
                insights.append(f"- 📈 Upward trend detected (slope: {trend:.4f})")
                insights.append("- This indicates improvement or growth over time")
            elif trend < 0:
                insights.append(f"- 📉 Downward trend detected (slope: {trend:.4f})")
                insights.append("- This suggests decline or reduction over time")
            else:
                insights.append("- ➡️ Relatively stable pattern with no strong trend")
        
        elif plot_type == "categorical":
            col = data_used['column']
            unique_count = self.df[col].nunique()
            most_common = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "N/A"
            
            insights.append(f"📊 **Categorical Analysis for {col}:**")
            insights.append(f"- {unique_count} unique categories identified")
            insights.append(f"- Most common category: {most_common}")
            
        elif plot_type == "clustering":
            insights.append("🎯 **Clustering Analysis:**")
            insights.append("- Data points are grouped based on similarity")
            insights.append("- Different colors represent distinct clusters")
            insights.append("- Useful for identifying patterns and segments in green skilling initiatives")
        
        return "\n".join(insights)

class SustainabilityAnalyzer:
    """Specialized analyzer for sustainability metrics"""
    
    @staticmethod
    def calculate_sustainability_score(df, sustainability_cols):
        """Calculate overall sustainability score"""
        if not sustainability_cols:
            return None
        
        scores = []
        for col in sustainability_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Normalize to 0-100 scale
                normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100
                scores.append(normalized.mean())
        
        return np.mean(scores) if scores else None
    
    @staticmethod
    def identify_sdg_alignment(df):
        """Identify alignment with UN Sustainable Development Goals"""
        sdg_keywords = {
            'SDG 4 (Quality Education)': ['education', 'learning', 'skill', 'training', 'qualification'],
            'SDG 7 (Affordable Clean Energy)': ['solar', 'wind', 'renewable', 'energy', 'clean'],
            'SDG 8 (Decent Work)': ['employment', 'job', 'work', 'career', 'occupation'],
            'SDG 9 (Industry Innovation)': ['innovation', 'technology', 'infrastructure', 'research'],
            'SDG 11 (Sustainable Cities)': ['urban', 'city', 'transport', 'housing', 'waste'],
            'SDG 12 (Responsible Consumption)': ['consumption', 'production', 'waste', 'recycling'],
            'SDG 13 (Climate Action)': ['climate', 'carbon', 'emission', 'greenhouse', 'temperature']
        }
        
        alignments = {}
        columns_text = ' '.join(df.columns).lower()
        
        for sdg, keywords in sdg_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in columns_text)
            if matches > 0:
                alignments[sdg] = matches
        
        return alignments

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_distribution_plots(df, analyzer):
    """Create distribution plots for numeric columns"""
    st.subheader("📊 Distribution Analysis")
    
    if not analyzer.numeric_cols:
        st.warning("No numeric columns found for distribution analysis.")
        return
    
    col1, col2 = st.columns(2)
    
    for idx, col in enumerate(analyzer.numeric_cols[:6]):  # Limit to 6 plots
        with col1 if idx % 2 == 0 else col2:
            # Histogram with KDE
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df[col].dropna(),
                name='Distribution',
                nbinsx=30,
                marker_color='#4caf50',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f'Distribution of {col}',
                xaxis_title=col,
                yaxis_title='Frequency',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate and display insights
            insights = analyzer.generate_insights("distribution", {'column': col})
            st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

def create_correlation_heatmap(df, analyzer):
    """Create correlation heatmap"""
    st.subheader("🔗 Correlation Analysis")
    
    if len(analyzer.numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[analyzer.numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        title='Correlation Heatmap of Numeric Variables'
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate insights
    insights = analyzer.generate_insights("correlation", {})
    st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)
    
    # Find strongest correlations
    st.subheader("💪 Strongest Correlations")
    
    # Get correlation pairs
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Variable 1': corr_matrix.columns[i],
                'Variable 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False).head(10)
    
    st.dataframe(corr_df, use_container_width=True)

def create_time_series_plots(df, analyzer):
    """Create time series visualizations"""
    st.subheader("📈 Temporal Analysis")
    
    detected = analyzer.detect_column_types()
    temporal_cols = detected['temporal_cols']
    
    if not temporal_cols:
        st.info("No temporal columns detected. Trying to use index as time dimension.")
        df['Index'] = range(len(df))
        temporal_cols = ['Index']
    
    if analyzer.numeric_cols:
        time_col = st.selectbox("Select time column:", temporal_cols)
        value_cols = st.multiselect("Select value columns:", analyzer.numeric_cols, 
                                     default=analyzer.numeric_cols[:3])
        
        if value_cols:
            fig = go.Figure()
            
            for col in value_cols:
                fig.add_trace(go.Scatter(
                    x=df[time_col],
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title='Time Series Trends',
                xaxis_title=time_col,
                yaxis_title='Value',
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights for first selected column
            if value_cols:
                insights = analyzer.generate_insights("time_series", {'column': value_cols[0]})
                st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

def create_categorical_analysis(df, analyzer):
    """Create categorical data visualizations"""
    st.subheader("🏷️ Categorical Analysis")
    
    if not analyzer.categorical_cols:
        st.warning("No categorical columns found.")
        return
    
    for col in analyzer.categorical_cols[:3]:  # Limit to 3
        value_counts = df[col].value_counts().head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig_bar = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={'x': col, 'y': 'Count'},
                title=f'Top Categories in {col}',
                color=value_counts.values,
                color_continuous_scale='Greens'
            )
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart
            fig_pie = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution of {col}',
                color_discrete_sequence=px.colors.sequential.Greens
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Insights
        insights = analyzer.generate_insights("categorical", {'column': col})
        st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

def create_advanced_analytics(df, analyzer):
    """Create advanced analytics visualizations"""
    st.subheader("🎯 Advanced Analytics")
    
    tabs = st.tabs(["PCA Analysis", "Clustering", "Statistical Tests", "Outlier Detection"])
    
    with tabs[0]:
        # PCA Analysis
        if len(analyzer.numeric_cols) >= 2:
            st.markdown("#### Principal Component Analysis")
            
            # Prepare data
            X = df[analyzer.numeric_cols].fillna(df[analyzer.numeric_cols].mean())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=min(3, len(analyzer.numeric_cols)))
            X_pca = pca.fit_transform(X_scaled)
            
            # Create DataFrame
            pca_df = pd.DataFrame(
                X_pca,
                columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
            )
            
            # 2D Plot
            if X_pca.shape[1] >= 2:
                fig = px.scatter(
                    pca_df, x='PC1', y='PC2',
                    title='PCA: First Two Principal Components',
                    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                           'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
                    color_discrete_sequence=['#4caf50']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Variance explained
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'Variance Explained': pca.explained_variance_ratio_ * 100
            })
            
            fig_var = px.bar(
                variance_df, x='Component', y='Variance Explained',
                title='Variance Explained by Each Component',
                color='Variance Explained',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_var, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            📊 <b>PCA Insights:</b><br>
            - PCA reduces dimensionality while preserving variance<br>
            - First few components capture most of the data variation<br>
            - Useful for identifying underlying patterns in green skilling data
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Need at least 2 numeric columns for PCA analysis.")
    
    with tabs[1]:
        # Clustering Analysis
        if len(analyzer.numeric_cols) >= 2:
            st.markdown("#### K-Means Clustering")
            
            n_clusters = st.slider("Number of clusters:", 2, 8, 3)
            
            # Prepare data
            X = df[analyzer.numeric_cols].fillna(df[analyzer.numeric_cols].mean())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create scatter plot
            cluster_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': clusters.astype(str)
            })
            
            fig = px.scatter(
                cluster_df, x='PC1', y='PC2', color='Cluster',
                title=f'K-Means Clustering (k={n_clusters})',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster sizes
            cluster_sizes = pd.Series(clusters).value_counts().sort_index()
            fig_sizes = px.bar(
                x=cluster_sizes.index.astype(str),
                y=cluster_sizes.values,
                labels={'x': 'Cluster', 'y': 'Count'},
                title='Cluster Sizes',
                color=cluster_sizes.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_sizes, use_container_width=True)
            
            insights = analyzer.generate_insights("clustering", {})
            st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)
        else:
            st.info("Need at least 2 numeric columns for clustering analysis.")
    
    with tabs[2]:
        # Statistical Tests
        st.markdown("#### Statistical Hypothesis Testing")
        
        if len(analyzer.numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                var1 = st.selectbox("Variable 1:", analyzer.numeric_cols, key='stat1')
            with col2:
                var2 = st.selectbox("Variable 2:", 
                                   [c for c in analyzer.numeric_cols if c != var1], 
                                   key='stat2')
            
            if st.button("Run Tests"):
                # T-test
                t_stat, p_value_t = stats.ttest_ind(
                    df[var1].dropna(), 
                    df[var2].dropna()
                )
                
                # Correlation test
                corr, p_value_corr = stats.pearsonr(
                    df[[var1, var2]].dropna()[var1],
                    df[[var1, var2]].dropna()[var2]
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("T-Test Statistic", f"{t_stat:.4f}")
                    st.metric("P-Value", f"{p_value_t:.4f}")
                    if p_value_t < 0.05:
                        st.success("✅ Statistically significant difference (p < 0.05)")
                    else:
                        st.info("ℹ️ No significant difference (p >= 0.05)")
                
                with col2:
                    st.metric("Correlation Coefficient", f"{corr:.4f}")
                    st.metric("P-Value", f"{p_value_corr:.4f}")
                    if abs(corr) > 0.7:
                        st.success("✅ Strong correlation")
                    elif abs(corr) > 0.3:
                        st.info("ℹ️ Moderate correlation")
                    else:
                        st.warning("⚠️ Weak correlation")
        else:
            st.info("Need at least 2 numeric columns for statistical tests.")
    
    with tabs[3]:
        # Outlier Detection
        st.markdown("#### Outlier Detection")
        
        if analyzer.numeric_cols:
            selected_col = st.selectbox("Select column for outlier detection:", 
                                       analyzer.numeric_cols)
            
            data = df[selected_col].dropna()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Box plot
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=data,
                name=selected_col,
                marker_color='#4caf50',
                boxmean='sd'
            ))
            
            fig.update_layout(
                title=f'Box Plot: {selected_col}',
                yaxis_title='Value',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Values", len(data))
            with col2:
                st.metric("Outliers Detected", len(outliers))
            with col3:
                st.metric("Outlier %", f"{len(outliers)/len(data)*100:.2f}%")
            
            if len(outliers) > 0:
                st.markdown("""
                <div class="insight-box">
                ⚠️ <b>Outlier Analysis:</b><br>
                - Outliers detected using IQR method (1.5 × IQR)<br>
                - These values fall significantly outside the typical range<br>
                - Consider investigating these cases for data quality or special circumstances
                </div>
                """, unsafe_allow_html=True)

def create_sustainability_dashboard(df, analyzer):
    """Create sustainability-specific dashboard"""
    st.subheader("🌱 Sustainability Metrics Dashboard")
    
    detected = analyzer.detect_column_types()
    sustainability_cols = detected['sustainability_cols']
    skill_cols = detected['skill_cols']
    
    if not sustainability_cols and not skill_cols:
        st.info("No sustainability or skill-related columns automatically detected. Analyzing all numeric data.")
        sustainability_cols = analyzer.numeric_cols[:5]
    
    # Calculate sustainability score
    sus_analyzer = SustainabilityAnalyzer()
    score = sus_analyzer.calculate_sustainability_score(df, sustainability_cols)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sustainability Score", f"{score:.1f}/100" if score else "N/A", 
                 delta="Good" if score and score > 70 else "Needs Improvement")
    
    with col2:
        st.metric("Total Records", f"{len(df):,}")
    
    with col3:
        st.metric("Data Completeness", 
                 f"{(1 - df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.1f}%")
    
    with col4:
        sdg_count = len(sus_analyzer.identify_sdg_alignment(df))
        st.metric("SDG Alignment", f"{sdg_count} Goals")
    
    # SDG Alignment
    st.markdown("#### 🎯 UN Sustainable Development Goals Alignment")
    
    sdg_alignments = sus_analyzer.identify_sdg_alignment(df)
    
    if sdg_alignments:
        sdg_df = pd.DataFrame(list(sdg_alignments.items()), 
                             columns=['SDG', 'Relevance Score'])
        sdg_df = sdg_df.sort_values('Relevance Score', ascending=False)
        
        fig = px.bar(
            sdg_df, x='Relevance Score', y='SDG',
            orientation='h',
            title='SDG Alignment Analysis',
            color='Relevance Score',
            color_continuous_scale='Greens',
            labels={'Relevance Score': 'Keyword Matches'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        🎯 <b>SDG Alignment Insights:</b><br>
        - Your data aligns with multiple UN Sustainable Development Goals<br>
        - Focus areas identified based on data column analysis<br>
        - Higher scores indicate stronger thematic alignment
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No specific SDG alignment detected. Upload data with sustainability-related columns for detailed analysis.")
    
    # Green Skills Analysis
    if skill_cols:
        st.markdown("#### 🎓 Green Skills Distribution")
        
        for col in skill_cols[:2]:
            if col in df.columns:
                value_counts = df[col].value_counts().head(10)
                
                fig = px.treemap(
                    names=value_counts.index,
                    parents=[""] * len(value_counts),
                    values=value_counts.values,
                    title=f'Green Skills: {col}',
                    color=value_counts.values,
                    color_continuous_scale='Greens'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🌱 Green Skilling Sustainable Goals Analytics")
    st.markdown("""
    Upload your CSV data and get comprehensive automated analysis with AI-powered insights
    for sustainable development and green skilling initiatives.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        This app automatically analyzes your data and generates:
        - 📊 Distribution analysis
        - 🔗 Correlation matrices
        - 📈 Time series trends
        - 🏷️ Categorical breakdowns
        - 🎯 Advanced analytics
        - 🌱 Sustainability metrics
        - 💡 AI-powered insights
        """)
        
        st.markdown("---")
        st.header("📚 Sample Data")
        if st.button("Generate Sample Green Skilling Data"):
            sample_df = pd.DataFrame({
                'Country': np.random.choice(['USA', 'Germany', 'India', 'China', 'Brazil'], 100),
                'Year': np.random.choice([2020, 2021, 2022, 2023, 2024], 100),
                'Green_Skills_Training_Hours': np.random.randint(10, 500, 100),
                'Solar_Energy_Capacity_MW': np.random.randint(100, 5000, 100),
                'Carbon_Emissions_Reduction_%': np.random.uniform(5, 40, 100),
                'Renewable_Energy_Jobs': np.random.randint(1000, 50000, 100),
                'Sustainability_Score': np.random.uniform(50, 95, 100),
                'Training_Program': np.random.choice(['Solar Installation', 'Wind Energy', 'Energy Efficiency', 'Green Building', 'Waste Management'], 100),
                'Employment_Rate_%': np.random.uniform(60, 95, 100),
                'Investment_USD_Million': np.random.uniform(5, 200, 100)
            })
            
            # Save to session state
            st.session_state['sample_data'] = sample_df
            st.success("✅ Sample data generated! It will be loaded automatically.")
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Initialize analyzer
            analyzer = DataAnalyzer(df)
            
            # Data Preview Section
            with st.expander("👁️ Data Preview", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(df.head(10), use_container_width=True)
                
                with col2:
                    summary = analyzer.get_summary_statistics()
                    st.markdown("### 📊 Quick Stats")
                    for key, value in summary.items():
                        st.metric(key.replace('_', ' ').title(), value)
            
            # Column Information
            with st.expander("📋 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null': df.isnull().sum().values,
                    'Unique': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                
                detected = analyzer.detect_column_types()
                
                st.markdown("#### 🔍 Auto-Detected Column Categories")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.success(f"**Sustainability**: {len(detected['sustainability_cols'])}")
                    if detected['sustainability_cols']:
                        st.write(detected['sustainability_cols'])
                
                with col2:
                    st.info(f"**Skills**: {len(detected['skill_cols'])}")
                    if detected['skill_cols']:
                        st.write(detected['skill_cols'])
                
                with col3:
                    st.warning(f"**Temporal**: {len(detected['temporal_cols'])}")
                    if detected['temporal_cols']:
                        st.write(detected['temporal_cols'])
                
                with col4:
                    st.error(f"**Geographic**: {len(detected['geographic_cols'])}")
                    if detected['geographic_cols']:
                        st.write(detected['geographic_cols'])
            
            # Main Analysis Tabs
            st.markdown("---")
            st.header("📊 Comprehensive Data Analysis")
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🌱 Sustainability Dashboard",
                "📊 Distributions", 
                "🔗 Correlations",
                "📈 Time Series",
                "🏷️ Categories",
                "🎯 Advanced Analytics"
            ])
            
            with tab1:
                create_sustainability_dashboard(df, analyzer)
            
            with tab2:
                create_distribution_plots(df, analyzer)
            
            with tab3:
                create_correlation_heatmap(df, analyzer)
            
            with tab4:
                create_time_series_plots(df, analyzer)
            
            with tab5:
                create_categorical_analysis(df, analyzer)
            
            with tab6:
                create_advanced_analytics(df, analyzer)
            
            # Overall Insights and Recommendations
            st.markdown("---")
            st.header("💡 Overall Insights & Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Key Findings")
                
                findings = []
                
                # Data quality finding
                completeness = (1 - df.isnull().sum().sum()/(len(df)*len(df.columns)))*100
                if completeness > 90:
                    findings.append("✅ Excellent data quality with minimal missing values")
                elif completeness > 70:
                    findings.append("⚠️ Good data quality, some cleaning may improve analysis")
                else:
                    findings.append("❗ Significant missing data detected, consider data cleaning")
                
                # Numeric analysis finding
                if analyzer.numeric_cols:
                    avg_skewness = df[analyzer.numeric_cols].skew().mean()
                    if abs(avg_skewness) < 0.5:
                        findings.append("✅ Data distributions are relatively balanced")
                    else:
                        findings.append("📊 Some variables show skewed distributions")
                
                # Correlation finding
                if len(analyzer.numeric_cols) >= 2:
                    corr_matrix = df[analyzer.numeric_cols].corr()
                    high_corr = (corr_matrix.abs() > 0.7).sum().sum() - len(corr_matrix)
                    if high_corr > 0:
                        findings.append(f"🔗 {high_corr} strong correlations detected between variables")
                
                # Categorical finding
                if analyzer.categorical_cols:
                    findings.append(f"🏷️ {len(analyzer.categorical_cols)} categorical dimensions for segmentation")
                
                # SDG alignment
                sdg_alignments = SustainabilityAnalyzer.identify_sdg_alignment(df)
                if sdg_alignments:
                    findings.append(f"🎯 Aligned with {len(sdg_alignments)} UN Sustainable Development Goals")
                
                for finding in findings:
                    st.markdown(f'<div class="insight-box">{finding}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🚀 Recommendations")
                
                recommendations = []
                
                # Data-driven recommendations
                if df.isnull().sum().sum() > 0:
                    recommendations.append("🔧 **Data Cleaning**: Address missing values for more accurate analysis")
                
                if analyzer.numeric_cols and any(df[col].std() > df[col].mean() * 2 for col in analyzer.numeric_cols[:3]):
                    recommendations.append("📊 **Outlier Investigation**: Some variables show high variability")
                
                if len(analyzer.numeric_cols) >= 3:
                    recommendations.append("🎯 **Clustering Analysis**: Consider segmenting data for targeted strategies")
                
                detected = analyzer.detect_column_types()
                if detected['temporal_cols']:
                    recommendations.append("📈 **Trend Analysis**: Monitor temporal patterns for forecasting")
                
                if detected['sustainability_cols'] or detected['skill_cols']:
                    recommendations.append("🌱 **Green Skills Focus**: Leverage sustainability metrics for impact assessment")
                
                recommendations.append("📚 **Continuous Monitoring**: Establish regular data collection and analysis cycles")
                recommendations.append("🤝 **Stakeholder Engagement**: Share insights with relevant stakeholders")
                
                for rec in recommendations:
                    st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
            
            # Executive Summary
            st.markdown("---")
            st.header("📝 Executive Summary")
            
            summary_text = f"""
            ### Data Overview
            - **Total Records**: {len(df):,}
            - **Variables**: {len(df.columns)}
            - **Data Completeness**: {completeness:.1f}%
            - **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            
            ### Sustainability Assessment
            """
            
            sus_score = SustainabilityAnalyzer.calculate_sustainability_score(df, detected['sustainability_cols'])
            if sus_score:
                summary_text += f"- **Sustainability Score**: {sus_score:.1f}/100\n"
            
            if sdg_alignments:
                summary_text += f"- **SDG Alignment**: {len(sdg_alignments)} goals identified\n"
                summary_text += f"- **Primary Focus Areas**: {', '.join(list(sdg_alignments.keys())[:3])}\n"
            
            summary_text += f"""
            
            ### Key Metrics
            """
            
            if analyzer.numeric_cols:
                for col in analyzer.numeric_cols[:5]:
                    mean_val = df[col].mean()
                    summary_text += f"- **{col}**: Mean = {mean_val:.2f}, Std = {df[col].std():.2f}\n"
            
            summary_text += """
            
            ### Data Quality
            """
            
            if completeness > 90:
                summary_text += "- ✅ **Excellent**: Minimal data quality issues detected\n"
            elif completeness > 70:
                summary_text += "- ⚠️ **Good**: Minor data quality improvements recommended\n"
            else:
                summary_text += "- ❗ **Needs Attention**: Significant data quality issues present\n"
            
            summary_text += f"""
            - **Missing Values**: {df.isnull().sum().sum()} cells ({(df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.2f}%)
            - **Duplicate Records**: {df.duplicated().sum()}
            
            ### Actionable Insights
            1. **Green Skilling Investment**: Focus on areas with highest sustainability scores
            2. **Training Optimization**: Analyze temporal trends to optimize training schedules
            3. **Regional Strategy**: Use geographic analysis for localized interventions
            4. **Performance Tracking**: Establish KPIs based on identified correlations
            5. **Stakeholder Reporting**: Use visualizations for transparent communication
            
            ### Next Steps
            1. Implement recommended data cleaning procedures
            2. Establish regular monitoring dashboards
            3. Conduct deeper analysis on high-correlation variables
            4. Develop predictive models for trend forecasting
            5. Create targeted action plans based on segmentation analysis
            """
            
            st.markdown(summary_text)
            
            # Download Report
            st.markdown("---")
            st.header("📥 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download cleaned data
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Cleaned Data (CSV)",
                    data=csv,
                    file_name="green_skilling_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download summary statistics
                summary_df = df.describe()
                summary_csv = summary_df.to_csv()
                st.download_button(
                    label="📈 Download Statistics (CSV)",
                    data=summary_csv,
                    file_name="statistics_summary.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Download executive summary
                st.download_button(
                    label="📝 Download Summary Report (TXT)",
                    data=summary_text,
                    file_name="executive_summary.txt",
                    mime="text/plain"
                )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted.")
    
    elif 'sample_data' in st.session_state:
        # Use sample data if generated
        df = st.session_state['sample_data']
        st.info("📊 Using generated sample data. Upload your own CSV file to analyze custom data.")
        
        # Initialize analyzer
        analyzer = DataAnalyzer(df)
        
        # Data Preview
        with st.expander("👁️ Sample Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Quick analysis
        st.markdown("---")
        st.header("📊 Quick Sample Analysis")
        
        tab1, tab2, tab3 = st.tabs(["🌱 Sustainability", "📊 Distributions", "🔗 Correlations"])
        
        with tab1:
            create_sustainability_dashboard(df, analyzer)
        
        with tab2:
            create_distribution_plots(df, analyzer)
        
        with tab3:
            create_correlation_heatmap(df, analyzer)
    
    else:
        # Welcome screen
        st.markdown("---")
        st.info("👆 Please upload a CSV file to begin analysis or generate sample data from the sidebar.")
        
        st.markdown("""
        ### 🌟 Features of this App
        
        This comprehensive analytics platform provides:
        
        #### 📊 Automated Visualizations
        - **Distribution Analysis**: Understand the spread and shape of your data
        - **Correlation Matrices**: Identify relationships between variables
        - **Time Series Trends**: Track changes over time
        - **Categorical Breakdowns**: Analyze segments and categories
        
        #### 🎯 Advanced Analytics
        - **PCA Analysis**: Reduce dimensionality and find patterns
        - **Clustering**: Segment your data intelligently
        - **Statistical Tests**: Validate hypotheses with rigorous testing
        - **Outlier Detection**: Identify anomalies in your data
        
        #### 🌱 Sustainability Focus
        - **SDG Alignment**: Automatic mapping to UN Sustainable Development Goals
        - **Green Skills Analysis**: Track and analyze skill development
        - **Sustainability Scoring**: Quantify sustainability performance
        - **Impact Assessment**: Measure and visualize impact metrics
        
        #### 💡 AI-Powered Insights
        - Automatic interpretation of every visualization
        - Context-aware recommendations
        - Data quality assessments
        - Actionable next steps
        
        ### 📚 How to Use
        
        1. **Upload Data**: Click "Browse files" in the sidebar and select your CSV file
        2. **Explore**: Navigate through different analysis tabs
        3. **Insights**: Read AI-generated insights for each visualization
        4. **Export**: Download reports and cleaned data
        
        ### 🎓 Perfect for
        
        - **Education Institutions**: Track green skilling programs
        - **Government Agencies**: Monitor sustainability initiatives
        - **NGOs**: Measure impact of environmental projects
        - **Corporations**: Assess ESG performance
        - **Researchers**: Analyze sustainability data
        
        ### 🚀 Get Started
        
        Generate sample data from the sidebar to see the app in action!
        """)
        
        # Example data structure
        st.markdown("---")
        st.subheader("📋 Example Data Structure")
        
        example_df = pd.DataFrame({
            'Country': ['USA', 'Germany', 'India'],
            'Year': [2023, 2023, 2023],
            'Green_Skills_Training_Hours': [250, 300, 180],
            'Solar_Energy_Capacity_MW': [3500, 2800, 4200],
            'Carbon_Emissions_Reduction_%': [25, 30, 20],
            'Renewable_Energy_Jobs': [25000, 18000, 35000],
            'Sustainability_Score': [78, 85, 72],
            'Training_Program': ['Solar Installation', 'Wind Energy', 'Energy Efficiency']
        })
        
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        Your CSV can include:
        - **Sustainability metrics**: Carbon emissions, renewable energy, etc.
        - **Skills data**: Training hours, certifications, programs
        - **Geographic data**: Countries, regions, cities
        - **Temporal data**: Years, dates, time periods
        - **Performance metrics**: Scores, ratings, KPIs
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌱 <b>Green Skilling Sustainable Goals Analytics Platform</b></p>
        <p>Empowering sustainable development through data-driven insights</p>
        <p>Made with ❤️ for a greener future | Powered by Streamlit & AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()