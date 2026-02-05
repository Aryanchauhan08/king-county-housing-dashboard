import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import folium
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------------------------------------------
# --- 1. Global Configuration ---
st.set_page_config(
    page_title="King County Housing Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional & Attractive Look
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #3498db;
    }
    
    /* Buttons in Grid */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        border: 1px solid #dfe6e9;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        border-color: #3498db;
        color: #3498db;
        background-color: #ecf0f1;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f4f8;
        color: #2980b9;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA ENGINE
# -----------------------------------------------------------------------------
# --- 2. Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    
    # Feature Engineering
    if "price" in df.columns and "sqft_living" in df.columns:
        df["price_per_sqft"] = df["price"] / df["sqft_living"]
        
    # Clean Data (Remove 0 prices)
    if "price" in df.columns:
        df = df[df["price"] > 0]

    # Clean city names if necessary (e.g. title case)
    if 'city' in df.columns:
        df['city'] = df['city'].apply(lambda x: str(x).title())
    return df

@st.cache_data
def load_geojson():
    try:
        with open('City_Boundaries.geojson') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return None

df = load_data()
geojson_data = load_geojson()

if df.empty:
    st.stop()

# -----------------------------------------------------------------------------
# GLOBAL SIDEBAR (Dynamic Controls for ALL Columns)
# -----------------------------------------------------------------------------
def reset_filters():
    st.session_state.clear()

with st.sidebar:
    st.header("🔍 Filter Controls")
    if st.button("🔄 Reset All Filters", on_click=reset_filters, use_container_width=True):
        st.rerun()
    
    st.sidebar.info("Filter data across all dimensions.")

    filters = {}
    filtered_df = df.copy()

    with st.form("global_filters"):
        # Group columns by type for better UX
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Critical Filters (Always visible)
        st.markdown("### 🔑 Key Filters")
        if 'price' in df.columns:
            min_p, max_p = int(df.price.min()), int(df.price.max())
            price_range = st.slider("Price Range", min_p, max_p, (min_p, max_p))
            filtered_df = filtered_df[filtered_df['price'].between(price_range[0], price_range[1])]

        if 'city' in df.columns:
            cities = sorted(df['city'].unique())
            sel_cities = st.multiselect("Cities", cities, default=[])
            if sel_cities:
                filtered_df = filtered_df[filtered_df['city'].isin(sel_cities)]

        # 2. Dynamic Expanders for everything else
        with st.expander("📏 Dimensions & specs"):
            for col in ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'floors']:
                if col in df.columns:
                    min_v, max_v = float(df[col].min()), float(df[col].max())
                    step = 1.0 if df[col].dtype == 'int64' else 0.1
                    # Use columns to save space
                    rng = st.slider(f"{col.replace('_', ' ').title()}", min_v, max_v, (min_v, max_v), step=step)
                    filtered_df = filtered_df[filtered_df[col].between(rng[0], rng[1])]

        with st.expander("✨ Quality & Features"):
            for col in ['condition', 'view', 'grade', 'waterfront']:
                if col in df.columns:
                    vals = sorted(df[col].unique())
                    sel = st.multiselect(f"{col.title()}", vals, default=[])
                    if sel:
                        filtered_df = filtered_df[filtered_df[col].isin(sel)]
        
        with st.expander("📅 Age & Renovation"):
            if 'yr_built' in df.columns:
                 min_y, max_y = int(df.yr_built.min()), int(df.yr_built.max())
                 y_rng = st.slider("Year Built", min_y, max_y, (min_y, max_y))
                 filtered_df = filtered_df[filtered_df['yr_built'].between(y_rng[0], y_rng[1])]
                 
        st.form_submit_button("Apply Filters")

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Records", f"{len(filtered_df):,}", f"{len(filtered_df)/len(df)*100:.1f}% of total")

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------
# --- 3. Main Tabs ---
tab_home, tab_uni, tab_bi, tab_multi, tab_geo, tab_summary, tab_raw = st.tabs([
    "🏠 Home", "📊 Univariate", "📈 Bivariate", "🔥 Multivariate", "🗺️ Geospatial", "📝 Summary", "💾 Raw Data"
])

# ====================
# TAB 1: HOME
# ====================
with tab_home:
    st.title("🏡 King County Housing Market Overview")
    
    # Summary Bullets
    st.info("""
    **Executive Summary:**
    *   **Market Trends:** Analyze weekly average prices to identify seasonal patterns.
    *   **Key Drivers:** Explore how square footage, location, and condition impact value.
    *   **Inventory:** Review the distribution of properties across different conditions and views.
    """)
    
    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Sales Price", f"${filtered_df.price.mean():,.0f}")
    m2.metric("Median Sales Price", f"${filtered_df.price.median():,.0f}")
    m3.metric("Total Volume", f"{len(filtered_df):,}")
    if 'sqft_living' in filtered_df.columns:
        m4.metric("Avg Price/SqFt", f"${(filtered_df.price / filtered_df.sqft_living).mean():.2f}")
    
    st.markdown("---")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### 📈 Market Price Trend")
        if 'date' in filtered_df.columns:
            daily = filtered_df.set_index('date').resample('W')['price'].mean().reset_index()
            fig = px.line(daily, x='date', y='price', markers=True, 
                         color_discrete_sequence=['#3498db'], 
                         title="Weekly Average Price Movement")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Text Analysis
            if len(daily) > 1:
                trend_diff = daily.iloc[-1]['price'] - daily.iloc[0]['price']
                trend_pct = (trend_diff / daily.iloc[0]['price']) * 100
                st.success(f"💡 **Analysis:** Over the selected period, prices have moved by **{trend_pct:+.1f}%** (${trend_diff:+,.0f}).")
            else:
                st.info("Not enough data points to show a trend.")
            
    with c2:
        st.markdown("#### 🏆 Key Statistics")
        st.markdown(f"""
        - **Highest Sale:** ${filtered_df.price.max():,.0f}
        - **Lowest Sale:** ${filtered_df.price.min():,.0f}
        - **Avg Bedroom Count:** {filtered_df.bedrooms.mean():.1f}
        - **Avg Year Built:** {int(filtered_df.yr_built.mean())}
        """)
        if 'condition' in filtered_df.columns:
            fig_pie = px.pie(filtered_df, names='condition', hole=0.5, title="Condition Breakdown",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

    # Row 3: View & Waterfront Pies
    st.markdown("### 🌊 Property Features Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        if 'view' in filtered_df.columns:
            fig_view = px.pie(filtered_df, names='view', title='Distribution of View Ratings', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
            fig_view.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_view, use_container_width=True)
    with c2:
        if 'waterfront' in filtered_df.columns:
            fig_water = px.pie(filtered_df, names='waterfront', title='Waterfront Properties (0=No, 1=Yes)', hole=0.4, color_discrete_sequence=px.colors.sequential.Blues)
            fig_water.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_water, use_container_width=True)

# ====================
# TAB 2: UNIVARIATE
# ====================
with tab_uni:
    st.subheader("🔎 Feature Distribution Analysis")
    st.markdown("Select a feature below to visualize its distribution and see key statistics.")
    
    # Selector instead of Grid
    cols_to_analyze = [c for c in df.columns if c not in ['date', 'street', 'statezip', 'country', 'price_per_sqft']]
    all_cols = filtered_df.columns.tolist()
    selected_uni_col = st.selectbox("Select Feature to Analyze:", cols_to_analyze, index=cols_to_analyze.index('bedrooms') if 'bedrooms' in cols_to_analyze else 0, key="uni_selector")
    
    # Visualization Area
    feature = selected_uni_col
    st.divider()
    
    c1, c2 = st.columns([3, 1])
    
    with c1:
        st.markdown(f"### Distribution of **{feature.title().replace('_', ' ')}**")
        
        is_cat = filtered_df[feature].dtype == 'object' or filtered_df[feature].nunique() < 15
        
        if is_cat:
            # Bar Chart / Count Plot
            counts = filtered_df[feature].value_counts().reset_index()
            counts.columns = [feature, 'count']
            fig = px.bar(counts, x=feature, y='count', color='count', 
                        color_continuous_scale='Blues', text_auto=True)
        else:
            # Histogram
            fig = px.histogram(filtered_df, x=feature, nbins=50, marginal="box", 
                              color_discrete_sequence=['#2ecc71'])
            
        fig.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("### 📝 Insights")
        stats = filtered_df[feature].describe()
        
        st.markdown(f"""
        **Key Insights for {feature.title().replace('_', ' ')}:**
        *   **Tyical Range:** Most properties fall between **{filtered_df[feature].quantile(0.25):,.0f}** and **{filtered_df[feature].quantile(0.75):,.0f}**.
        *   **Average:** The average value is **{stats['mean'] if 'mean' in stats else 'N/A':,.2f}**, while the middle-of-the-pack (median) value is **{filtered_df[feature].median() if np.issubdtype(filtered_df[feature].dtype, np.number) else 'N/A':,.2f}**.
        """)
        
        # Smart Analysis Text
        if not is_cat and np.issubdtype(filtered_df[feature].dtype, np.number):
            skew = filtered_df[feature].skew()
            skew_desc = "balanced"
            if skew > 1: skew_desc = "concentrated on the lower end, with some high-value outliers"
            if skew < -1: skew_desc = "concentrated on the higher end"
            st.info(f"💡 **Plain English:** The data is **{skew_desc}**. This means most homes have a lower {feature.replace('_', ' ')}, but a few 'luxury' extremes pull the average up.")
        elif is_cat:
            top_val = filtered_df[feature].mode()[0]
            st.info(f"💡 **Plain English:** The most popular category is **{top_val}**. This is the standard for this market.")

# ====================
# TAB 3: BIVARIATE
# ====================
with tab_bi:
    st.subheader("⚖️ Price Correlation Analysis")
    st.markdown("Compare any feature against **Price** to find correlations.")
    
    # Selector
    target_cols = [c for c in df.columns if c not in ['date', 'street', 'statezip', 'country', 'price', 'price_per_sqft']]
    selected_bi_col = st.selectbox("Select Feature to Compare with Price:", target_cols, key="bi_selector")
                
    feat = selected_bi_col
    st.divider()
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"### Price vs **{feat.title().replace('_', ' ')}**")
        
        is_num = np.issubdtype(filtered_df[feat].dtype, np.number) and filtered_df[feat].nunique() > 15
        
        if is_num:
            # Scatter
            fig = px.scatter(filtered_df, x=feat, y='price', color='condition', 
                            trendline='ols', opacity=0.6,
                            title=f"Correlation Scatter Plot",
                            color_continuous_scale='Viridis')
        else:
            # Box Plot
            fig = px.box(filtered_df, x=feat, y='price', color=feat, 
                        title=f"Price Distribution by Category")
            
        fig.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("### 📊 Correlation Analysis")
        corr_val = 0
        if is_num:
            corr = filtered_df[[feat, 'price']].corr().iloc[0,1]
            corr_val = corr
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            direction = "positive" if corr > 0 else "negative"
            
            st.metric(f"Correlation Strength", f"{strength.title()}")
    st.header("📈 Bivariate Analysis")
    st.markdown("Explore relationships between two variables to understand how they interact.")

    # --- numeric vs numeric (Scatter) ---
    st.subheader("Numeric vs. Numeric Correlation")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X-Axis", num_cols, index=num_cols.index("sqft_living") if "sqft_living" in num_cols else 0)
    with c2:
        y_axis = st.selectbox("Y-Axis", num_cols, index=num_cols.index("price") if "price" in num_cols else 0)
    
    fig_scat = px.scatter(filtered_df, x=x_axis, y=y_axis, color="price", title=f"{x_axis} vs {y_axis}",
                          color_continuous_scale="Viridis", opacity=0.6)
    st.plotly_chart(fig_scat, use_container_width=True)

    if x_axis == "sqft_living" and y_axis == "price":
         st.info("""
        **💡 Plain English Insight:**
        There is a clear **positive relationship** between **living area** and **price**: as homes get larger, they typically cost more. However, the spread of points widens for larger homes (a "fanning out" pattern), suggesting that for luxury properties, **locations** or **other amenities** play a huge role in price beyond just square footage.
        """)
    
    st.divider()

    # --- FULL WIDTH HEATMAP ---
    st.subheader("🔥 Correlation Heatmap")
   
    # Compute correlation
    corr_mat = filtered_df.select_dtypes(include=['number']).corr()
    
    # Create Heatmap
    fig_hm = px.imshow(
        corr_mat, 
        text_auto=".2f", 
        color_continuous_scale="BuGn",  # Using BuGn as requested
        zmin=-1, 
        zmax=1, 
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    
    # Insight for Heatmap
    st.success("""
    **💡 Key Insight:**
    The heatmap confirms that **price** is most strongly correlated with **sqft_living** and **sqft_above**, visually reinforcing that **property size is the primary driver of value**. The **view rating** also shows a **very high positive correlation**.
    """)

    st.divider()

    # --- FULL WIDTH TOP 20 CITIES ---
    st.subheader("🏙️ Top 20 Cities by Volume")
    
    # Top 20 Cities Logic
    if 'city' in filtered_df.columns:
        top_cities = filtered_df['city'].value_counts().nlargest(20).reset_index()
        top_cities.columns = ['city', 'count']
        
        # Bar Chart
        fig_city = px.bar(
            top_cities, 
            x='count', 
            y='city', 
            orientation='h',
            title="Top 20 Cities by Number of Houses",
            color='count',
            color_continuous_scale="Greens" # Matches the green theme
        )
        # Invert y-axis so top city is at the top
        fig_city.update_layout(yaxis=dict(autorange="reversed"))
        
        st.plotly_chart(fig_city, use_container_width=True)
        
        # Insight for Cities
        st.success("""
        **💡 Market Insight:**
        The chart clearly confirms that **Seattle** is the **primary real estate market** in this dataset, with a **house count far exceeding** all other cities. 
        There is a **steep drop-off** after the top few cities, with **Bellevue** and **Kent** forming a **clear second tier**, followed by another decline to cities like **Redmond** and **Renton**. 
        This visualization emphasizes the dataset's **hyper-local focus**, with the **top 20 cities** representing the **most active and significant markets**.
        """)
    else:
        st.info("City data not available for this chart.")

# ====================
# TAB 4: MULTIVARIATE
# ====================
with tab_multi:
    st.subheader("🔗 Complex Relationships")
    
    st.info("Explore complex interactions between 3 variables at once.")

    st.divider()
    st.subheader("3D Multivariate Explorer")
    
    # Define num_cols_only locally for this tab
    num_cols_only = filtered_df.select_dtypes(include=['number']).columns
    
    x = st.selectbox("X Axis", num_cols_only, index=list(num_cols_only).index('sqft_living') if 'sqft_living' in num_cols_only else 0)
    y = st.selectbox("Y Axis", ['price'])
    z = st.selectbox("Z Axis", num_cols_only, index=list(num_cols_only).index('yr_built') if 'yr_built' in num_cols_only else 0)
    c = st.selectbox("Color", filtered_df.columns, index=list(filtered_df.columns).index('grade') if 'grade' in filtered_df.columns else 0)
    
    fig_3d = px.scatter_3d(filtered_df, x=x, y=y, z=z, color=c, opacity=0.7, color_continuous_scale="Viridis")
    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

# ====================
# TAB 5: GEO MAP
# ====================
with tab_geo:
    st.subheader("Geospatial Intelligence")
    st.info("💡 **Insight:** Location is the most critical factor in real estate. The map below reveals how prices cluster near **waterfronts** and **city centers**, visualizing the 'location, location, location' adage.")
    
    if geojson_data and 'city' in filtered_df.columns:
        # --- Prepare Data for Tooltips ---
        # Aggregate data by city
        city_data = filtered_df.groupby('city').agg(
            average_price=('price', 'mean'),
            average_sqft_living=('sqft_living', 'mean')
        ).reset_index()
        
        # Create a dictionary for fast lookup
        city_metrics_dict = city_data.set_index('city').to_dict('index')
        
        # Inject metrics into GeoJSON properties for tooltips
        # Note: We act on a copy or modify in place if safe (Streamlit caches data, so be careful. 
        # Ideally we shouldn't mute cached objects, but for folium tooltip this is the standard way).
        # To avoid mutating shared state, we can assume geojson_data is loaded fresh or just do it.
        # Given Streamlit caching, it's safer to not rely on mutation persistence or deep copy if heavy.
        # But for this scope, let's update the features directly as per notebook logic.
        
        for feature in geojson_data['features']:
            city_name = feature['properties'].get('CITY_DISSOLVE')
            if city_name:
                metrics = city_metrics_dict.get(city_name, {})
                price = metrics.get('average_price')
                sqft = metrics.get('average_sqft_living')
                feature['properties']['avg_price_str'] = f"${price:,.0f}" if price else 'N/A'
                feature['properties']['avg_sqft_str'] = f"{int(sqft):,} sqft" if sqft else 'N/A'

        # --- Create Folium Map ---
        m = folium.Map(
            location=[47.50, -121.00], 
            zoom_start=9,
            tiles="cartodbpositron"
        )
        
        # --- Layer 1: Average Price (YlOrRd) ---
        price_layer = folium.Choropleth(
            geo_data=geojson_data,
            name='Average House Price',
            data=city_data,
            columns=['city', 'average_price'],
            key_on='feature.properties.CITY_DISSOLVE',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name='Average House Price ($)',
            highlight=True,
            nan_fill_color='white',
            nan_fill_opacity=0.4
        ).add_to(m)
        
        # Tooltip for Price Layer
        folium.features.GeoJsonTooltip(
            fields=['CITY_DISSOLVE', 'avg_price_str'],
            aliases=['City:', 'Avg. Price:'],
            localize=True,
            sticky=False,
            labels=True,
            style="background-color: #F0EFEF; border: 2px solid black; border-radius: 3px; box-shadow: 3px;"
        ).add_to(price_layer.geojson)

        # --- Layer 2: Average Sqft Living (GnBu) ---
        sqft_layer = folium.Choropleth(
            geo_data=geojson_data,
            name='Average Living Area (sqft)',
            data=city_data,
            columns=['city', 'average_sqft_living'],
            key_on='feature.properties.CITY_DISSOLVE',
            fill_color='GnBu',
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name='Average Living Area (sqft)',
            highlight=True,
            show=False, # Default hidden
            nan_fill_color='white',
            nan_fill_opacity=0.4
        ).add_to(m)
        
        # Tooltip for Sqft Layer
        folium.features.GeoJsonTooltip(
            fields=['CITY_DISSOLVE', 'avg_sqft_str'],
            aliases=['City:', 'Avg. Sqft Living:'],
            localize=True,
            sticky=False,
            labels=True,
            style="background-color: #F0EFEF; border: 2px solid black; border-radius: 3px; box-shadow: 3px;"
        ).add_to(sqft_layer.geojson)

        # Add Layer Control
        folium.LayerControl().add_to(m)

        # Render Map
        st_folium(m, width=1200, height=600)
        
    else:
        st.warning("GeoJSON data or City column missing.")

# --- TAB 6: SUMMARY ---
with tab_summary:
    st.markdown("""
    ## 📝 Executive Summary & Insights
    
    ### 🏡 Market Overview
    The King County housing market data reveals several defining characteristics about the region's properties.
    
    *   **Family-Oriented Market:** The dataset is strongly dominated by **3 and 4-bedroom homes**, indicating that the core market consists of mid-sized family residences. Studio apartments or extremely large estates (7+ bedrooms) are rare outliers.
    *   **Condition Matters:** Most homes are in **Average (3)** condition. However, a significant portion is rated as **Good (4)** or **Very Good (5)**, suggesting a generally well-maintained housing stock. Poor condition homes are very niche.
    
    ### 💰 Price Drivers
    *   **Square Footage Rule:** There is a clear, **strong positive relationship** between living space (sqft_living) and price. As size increases, price follows—often exponentially.
    *   **The "Bell Curve" of Size:** Most homes fall between **1,500 and 2,500 sqft**. This is the "sweet spot" of the market.
    *   **Bathrooms:** Standard homes typically feature **2.5 bathrooms**. More bathrooms are almost exclusively found in luxury, higher-sqft properties.
    
    ### 📍 Location & Features
    *   **Views & Waterfronts:** While rare, properties with waterfront access or significant views command huge premiums.
    *   **Basements:** Basements are less common than expected, often appearing only in specific architectural styles or larger homes.
    
    ### ✅ Conclusion
    For buyers, **mid-sized homes in average-to-good condition** offer the most inventory. For sellers, **improving condition** and **maximizing usable square footage** correspond directly to value increases. The market is healthy, with a solid baseline of standard family homes and a thriving high-end luxury segment.
    """)

# --- TAB 7: RAW DATA ---
with tab_raw:
    st.subheader("📄 Raw Data Explorer")
    st.markdown(f"Displaying **{len(filtered_df)}** records matching your filters.")
    st.dataframe(filtered_df, use_container_width=True)
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="filtered_housing_data.csv",
        mime="text/csv",
    )
