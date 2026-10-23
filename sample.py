import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Mobile Analytics & AI Recommender",
    page_icon="📱",
    layout="wide"
)

# Inject custom CSS for dark background styling
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PERMANENT MAIN PAGE TITLE (Guaranteed to show at the top)
# -------------------------------------------------------------
st.title("📱 Smartphone Market Analytics & AI Recommender System")
st.markdown("### *Interactive Market Segmentation, Performance Analytics & Similarity Discovery Dashboard*")
st.markdown("---")

# -------------------------------------------------------------
# DATA LOADING & MODEL CACHING
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_smartphones.csv")
    return df

data_df = load_data()

cluster_titles = {
    0: "Cluster 0: Budget & Value Volume Drivers",
    1: "Cluster 1: Premium Flagship Devices",
    2: "Cluster 2: High-Satisfaction Mid-Range",
    3: "Cluster 3: Niche & Problematic Models"
}
data_df['cluster_title'] = data_df['cluster'].map(cluster_titles)

@st.cache_resource
def train_recommender(df):
    feature_cols = ['price_usd', 'rating', 'camera_rating', 'performance_rating']
    X = df[feature_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    nn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
    nn_model.fit(X_scaled)
    return nn_model, X_scaled, scaler

model, X_scaled, scaler = train_recommender(data_df)

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.title("📱 Navigation Hub")
app_mode = st.sidebar.radio(
    "Select Module:", 
    ["📊 Granular Cluster EDA", "🤖 Similarity Recommender", "📈 Strategic Insights & Reporting"]
)

# -------------------------------------------------------------
# MODULE 1: GRANULAR CLUSTER EDA
# -------------------------------------------------------------
if app_mode == "📊 Granular Cluster EDA":
    st.header("📊 Granular Cluster-Specific EDA")
    st.markdown("Explore interactive market behaviors, brand dominance, and quality metrics with vibrant visual breakdowns.")
    
    selected_cluster_name = st.selectbox("Select Cluster Segment to Analyze:", options=list(cluster_titles.values()))
    cluster_id = [k for k, v in cluster_titles.items() if v == selected_cluster_name][0]
    filtered_df = data_df[data_df['cluster'] == cluster_id]
    
    st.markdown(f"""
        <div style="background-color: #1e293b; padding: 15px; border-radius: 10px; border-left: 5px solid #38bdf8; margin-bottom: 25px;">
            <h3 style="margin:0; color: #f8fafc;">Segment Overview: {selected_cluster_name}</h3>
            <p style="margin:5px 0 0 0; color: #94a3b8;">Total Devices in Segment: <b>{len(filtered_df)}</b></p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Top 5 Brands by Product Volume")
        top_brands = filtered_df['brand'].value_counts().head(5).reset_index()
        top_brands.columns = ['brand', 'count']
        
        fig_bar1 = px.bar(
            top_brands, x='brand', y='count', text='count', 
            color='count', color_continuous_scale='Sunset'
        )
        fig_bar1.update_traces(texttemplate='%{text}', textposition='outside', marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
        fig_bar1.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar1, use_container_width=True)
        
        st.markdown("### 🥧 Brand Market Share Breakdown")
        fig_pie1 = px.pie(
            top_brands, names='brand', values='count',
            hole=0.5, color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie1.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#0f172a', width=2)))
        fig_pie1.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie1, use_container_width=True)

    with col2:
        st.markdown("### 📊 Core Performance & Quality Scores")
        metrics_avg = pd.DataFrame({
            'Metric': ['Overall Rating', 'Camera Rating', 'Performance Rating'],
            'Score': [
                filtered_df['rating'].mean(),
                filtered_df['camera_rating'].mean(),
                filtered_df['performance_rating'].mean()
            ]
        })
        
        fig_bar2 = px.bar(
            metrics_avg, x='Metric', y='Score', 
            color='Score', color_continuous_scale='Tealgrn', 
            range_y=[0, 5], text='Score'
        )
        fig_bar2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar2.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar2, use_container_width=True)
        
        st.markdown("### 🥧 Regional / Category Distribution")
        cat_col = 'country' if 'country' in filtered_df.columns else 'brand'
        top_cats = filtered_df[cat_col].value_counts().head(4).reset_index()
        top_cats.columns = [cat_col, 'count']
        
        fig_pie2 = px.pie(
            top_cats, names=cat_col, values='count',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie2.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#0f172a', width=2)))
        fig_pie2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie2, use_container_width=True)

# -------------------------------------------------------------
# MODULE 2: SIMILARITY RECOMMENDATION SYSTEM
# -------------------------------------------------------------
elif app_mode == "🤖 Similarity Recommender":
    st.header("🤖 Memory-Efficient Similarity Recommender")
    st.markdown("Select a reference smartphone model to instantly discover similar alternatives based on optimized distance metrics.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        selected_brand = st.selectbox("Select Brand:", options=sorted(data_df['brand'].unique()))
    
    brand_filtered_df = data_df[data_df['brand'] == selected_brand]
    
    with col_b:
        selected_model_row = st.selectbox(
            "Select Specific Model:", 
            options=brand_filtered_df.index,
            format_func=lambda idx: f"{data_df.loc[idx, 'brand']} (Price: ${data_df.loc[idx, 'price_usd']} | Rating: {data_df.loc[idx, 'rating']})"
        )
    
    if st.button("🔍 Get Recommendations", type="primary"):
        target_vector = X_scaled[selected_model_row].reshape(1, -1)
        distances, indices = model.kneighbors(target_vector)
        
        rec_indices = indices[0][1:]
        rec_distances = distances[0][1:]
        
        recommendations = data_df.iloc[rec_indices].copy()
        recommendations['similarity_distance'] = rec_distances
        
        target_device = data_df.loc[selected_model_row]
        
        st.markdown("### 📌 Selected Target Device")
        st.success(f"**Brand:** {target_device['brand']} | **Price:** ${target_device['price_usd']} | **Rating:** {target_device['rating']} | **Segment:** {target_device['cluster_title']}")
        
        st.markdown("### ✨ Top 5 Recommended Alternatives")
        display_cols = ['brand', 'price_usd', 'rating', 'camera_rating', 'performance_rating', 'cluster_title', 'similarity_distance']
        st.dataframe(recommendations[display_cols].reset_index(drop=True), use_container_width=True)

# -------------------------------------------------------------
# MODULE 3: INSIGHTS & STRATEGIC REPORTING
# -------------------------------------------------------------
elif app_mode == "📈 Strategic Insights & Reporting":
    st.header("📈 Strategic Insights & Reporting")
    st.markdown("Actionable analytics on product segmentation, price-to-performance trends, and customer preference patterns.")
    
    st.subheader("💡 Price vs. Performance Trend Analysis")
    fig_scatter = px.scatter(
        data_df, x="price_usd", y="rating", color="cluster_title",
        hover_data=['brand', 'camera_rating', 'performance_rating'],
        title="Price vs. Overall Rating Across Market Segments",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_scatter.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 High-Performing Segments")
        st.success(
            "**Cluster 2 (High-Satisfaction Mid-Range):**\n"
            "- Delivers peak user ratings (averaging 4.38+).\n"
            "- Maximizes price-to-performance value, driving customer advocacy."
        )
        st.success(
            "**Cluster 1 (Premium Flagship):**\n"
            "- Anchored by luxury brands like Apple (~$1,022 average price).\n"
            "- Superior specs justify top-tier market positioning."
        )

    with col2:
        st.markdown("### ⚠️ Low-Performing / Problematic Segments")
        st.error(
            "**Cluster 3 (Niche / Problematic Tier):**\n"
            "- Suffers from severe rating drop-offs (averaging 1.71).\n"
            "- Signals specification or quality control gaps requiring review."
        )
        st.warning(
            "**Cluster 0 (Volume Drivers):**\n"
            "- Mass-market volume leaders operating at budget price points.\n"
            "- Highly price-sensitive with moderate performance baselines."
        )
    
    st.markdown("---")
    st.subheader("🎯 Executive Summary & Decision Making")
    st.info(
        "1. **Portfolio Focus:** Prioritize marketing and inventory acquisition around Cluster 2 mid-range devices.\n"
        "2. **Risk Mitigation:** Rectify or phase out models categorized under Cluster 3 to protect brand equity.\n"
        "3. **Platform Integration:** The memory-efficient K-NN distance model powers real-time product cross-selling without server bottlenecks."
    )