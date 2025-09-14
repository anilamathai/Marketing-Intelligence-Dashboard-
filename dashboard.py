import pandas as pd
import streamlit as st
import plotly.express as px


# --- Section 1: Data Loading ---
@st.cache_data
def load_data():
    """Loads all datasets from CSV files."""
    try:
        df_facebook = pd.read_csv("Facebook.csv")
        df_google = pd.read_csv("Google.csv")
        df_tiktok = pd.read_csv("TikTok.csv")
        df_business = pd.read_csv("business.csv")
        return df_facebook, df_google, df_tiktok, df_business
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure the data files are in the correct directory.")
        return None, None, None, None

# --- Section 2: Corrected Cleaning Functions ---
def clean_facebook_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes the Facebook DataFrame for analysis."""
    if not df.empty:
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
        df['platform'] = 'Facebook'
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        numeric_cols = ['impression', 'clicks', 'spend', 'attributed_revenue']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna("Unknown")
        if 'impression' in df.columns and 'clicks' in df.columns:
            df['ctr'] = df.apply(lambda x: x['clicks'] / x['impression'] if x['impression'] > 0 else 0, axis=1)
        if 'spend' in df.columns and 'attributed_revenue' in df.columns:
            df['roas'] = df.apply(lambda x: x['attributed_revenue'] / x['spend'] if x['spend'] > 0 else 0, axis=1)
        if 'spend' in df.columns and 'clicks' in df.columns:
            df['cpc'] = df.apply(lambda x: x['spend'] / x['clicks'] if x['clicks'] > 0 else 0, axis=1)
        df.drop_duplicates(inplace=True)
    return df

def clean_google_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes the Google Ads DataFrame for analysis."""
    if not df.empty:
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
        df['platform'] = 'Google'
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        numeric_cols = ['impression', 'clicks', 'spend', 'attributed_revenue']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna("Unknown")
        if 'impression' in df.columns and 'clicks' in df.columns:
            df['ctr'] = df.apply(lambda x: x['clicks'] / x['impression'] if x['impression'] > 0 else 0, axis=1)
        if 'spend' in df.columns and 'attributed_revenue' in df.columns:
            df['roas'] = df.apply(lambda x: x['attributed_revenue'] / x['spend'] if x['spend'] > 0 else 0, axis=1)
        if 'spend' in df.columns and 'clicks' in df.columns:
            df['cpc'] = df.apply(lambda x: x['spend'] / x['clicks'] if x['clicks'] > 0 else 0, axis=1)
        df.drop_duplicates(inplace=True)
    return df

def clean_tiktok_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes the TikTok Ads DataFrame for analysis."""
    if not df.empty:
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
        df['platform'] = 'TikTok'
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        numeric_cols = ['impression', 'clicks', 'spend', 'attributed_revenue']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna("Unknown")
        if 'impression' in df.columns and 'clicks' in df.columns:
            df['ctr'] = df.apply(lambda x: x['clicks'] / x['impression'] if x['impression'] > 0 else 0, axis=1)
        if 'spend' in df.columns and 'attributed_revenue' in df.columns:
            df['roas'] = df.apply(lambda x: x['attributed_revenue'] / x['spend'] if x['spend'] > 0 else 0, axis=1)
        if 'spend' in df.columns and 'clicks' in df.columns:
            df['cpc'] = df.apply(lambda x: x['spend'] / x['clicks'] if x['clicks'] > 0 else 0, axis=1)
        df.drop_duplicates(inplace=True)
    return df

def clean_business_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes the Business DataFrame."""
    if not df.empty:
        df.columns = (
            df.columns.str.lower()
            .str.replace(' ', '_')
            .str.replace('#_of_', '', regex=True)
            .str.strip()
        )
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        numeric_cols = ['orders', 'new_orders', 'new_customers', 'total_revenue', 'gross_profit', 'cogs']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna("Unknown")
        df.drop_duplicates(inplace=True)
    return df

# --- Section 3: Calling Functions and Joining Data ---
df_facebook, df_google, df_tiktok, df_business = load_data()

if df_facebook is not None and not df_facebook.empty:
    df_facebook_cleaned = clean_facebook_data(df_facebook)
    df_google_cleaned = clean_google_data(df_google)
    df_tiktok_cleaned = clean_tiktok_data(df_tiktok)
    df_business_cleaned = clean_business_data(df_business)

    combined_marketing_df = pd.concat(
        [df_facebook_cleaned, df_google_cleaned, df_tiktok_cleaned],
        ignore_index=True
    )

    master_df = pd.merge(
        combined_marketing_df,
        df_business_cleaned,
        on='date',
        how='left'
    )

    business_cols = ['orders', 'new_orders', 'new_customers', 'total_revenue', 'gross_profit', 'cogs']
    for col in business_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna(0)

    # --- Section 4: Aggregating Data and Deriving Metrics ---
    daily_marketing_performance = master_df.groupby(['date', 'platform']).agg(
        total_spend=('spend', 'sum'),
        total_impressions=('impression', 'sum'),
        total_clicks=('clicks', 'sum'),
        total_attributed_revenue=('attributed_revenue', 'sum')
    ).reset_index()

    daily_business_performance = df_business_cleaned.groupby('date').agg(
        total_orders=('orders', 'sum'),
        total_new_orders=('new_orders', 'sum'),
        total_new_customers=('new_customers', 'sum'),
        total_revenue=('total_revenue', 'sum'),
        total_gross_profit=('gross_profit', 'sum')
    ).reset_index()

    print("Data preparation complete. All steps were successful.")


    # --- Section 5: Dashboard Setup and Filters ---
    st.set_page_config(layout="wide", page_title="Marketing Intelligence Dashboard")
    st.title("🚀 Marketing Intelligence Dashboard")
    st.markdown("A unified view of our marketing efforts and their business impact, now with interactive filters.")

    # --- Sidebar Visibility Toggle ---
    if 'show_sidebar' not in st.session_state:
        st.session_state.show_sidebar = True

    col_toggle, _ = st.columns([0.15, 0.85])
    with col_toggle:
        if st.button("Toggle Filters"):
            st.session_state.show_sidebar = not st.session_state.show_sidebar
    
    st.markdown("---")

    if st.session_state.show_sidebar:
        with st.sidebar:
            st.header("Filter Options")
            
            # --- Organize and space out the filters ---
            st.subheader("Date Range")
            start_date = pd.to_datetime(master_df['date']).min()
            end_date = pd.to_datetime(master_df['date']).max()

            date_range = st.date_input(
                "Select a date range",
                value=(start_date, end_date),
                min_value=start_date,
                max_value=end_date,
                help="Choose a start and end date for the analysis."
            )

            # Add space below the date input
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.subheader("Platform Selection")
            available_platforms = master_df['platform'].unique().tolist()
            selected_platforms = st.multiselect(
                "Select marketing platforms",
                options=available_platforms,
                default=available_platforms,
                help="Choose one or more platforms to include in the dashboard."
            )

            if len(date_range) == 2:
                start_date_filter, end_date_filter = date_range
            else:
                start_date_filter, end_date_filter = start_date, end_date

    else:
        start_date_filter = pd.to_datetime(master_df['date']).min()
        end_date_filter = pd.to_datetime(master_df['date']).max()
        selected_platforms = master_df['platform'].unique().tolist()


    # --- Apply Filters to DataFrames ---
    filtered_df = master_df[
        (master_df['date'] >= pd.to_datetime(start_date_filter)) &
        (master_df['date'] <= pd.to_datetime(end_date_filter)) &
        (master_df['platform'].isin(selected_platforms))
    ]

    if not filtered_df.empty:
        daily_marketing_performance_filtered = filtered_df.groupby(['date', 'platform']).agg(
            total_spend=('spend', 'sum'),
            total_impressions=('impression', 'sum'),
            total_clicks=('clicks', 'sum'),
            total_attributed_revenue=('attributed_revenue', 'sum')
        ).reset_index()

        daily_business_performance_filtered = filtered_df.groupby('date').agg(
            total_orders=('orders', 'sum'),
            total_new_orders=('new_orders', 'sum'),
            total_new_customers=('new_customers', 'sum'),
            total_revenue=('total_revenue', 'sum'),
            total_gross_profit=('gross_profit', 'sum')
        ).reset_index()
    else:
        daily_marketing_performance_filtered = pd.DataFrame(columns=daily_marketing_performance.columns)
        daily_business_performance_filtered = pd.DataFrame(columns=daily_business_performance.columns)
        st.warning("No data available for the selected filters.")
        st.stop()

    # --- Main Dashboard Layout ---

    # KPI Section
    st.subheader("Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        total_spend = filtered_df['spend'].sum()
        st.metric(label="Total Ad Spend", value=f"${total_spend:,.2f}")
    with kpi_col2:
        total_attributed_revenue = filtered_df['attributed_revenue'].sum()
        st.metric(label="Total Attributed Revenue", value=f"${total_attributed_revenue:,.2f}")
    with kpi_col3:
        total_roas = total_attributed_revenue / total_spend if total_spend > 0 else 0
        st.metric(label="Overall ROAS", value=f"{total_roas:.2f}")
    with kpi_col4:
        total_new_customers = filtered_df['new_customers'].sum()
        st.metric(label="Total New Customers", value=f"{int(total_new_customers):,}")

    st.markdown("---")

    # Performance Trends (Row-wise Layout)
    st.subheader("Performance Trends")

    # Daily Business Performance Chart
    if not daily_business_performance_filtered.empty:
        daily_business_performance_melted = daily_business_performance_filtered.melt(
            id_vars=['date'],
            value_vars=['total_revenue', 'total_gross_profit'],
            var_name='Metric',
            value_name='Value'
        )
        fig_business_trend = px.line(
            daily_business_performance_melted,
            x="date",
            y="Value",
            color="Metric",
            title="Total Revenue vs. Gross Profit Over Time",
            labels={'Value': 'Amount ($)', 'date': 'Date'},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_business_trend.update_layout(legend_title="Metric", hovermode="x unified")
        st.plotly_chart(fig_business_trend, use_container_width=True)

    # Total Revenue vs. Total Spend Scatter Plot
    if not filtered_df.empty:
        daily_revenue_spend = filtered_df.groupby('date').agg(
            total_revenue=('total_revenue', 'sum'),
            total_spend=('spend', 'sum')
        ).reset_index()
        fig_scatter = px.scatter(
            daily_revenue_spend,
            x="total_spend",
            y="total_revenue",
            size="total_revenue",
            title="Daily Total Revenue vs. Total Ad Spend",
            labels={'total_spend': 'Total Ad Spend ($)', 'total_revenue': 'Total Revenue ($)'},
            hover_data=['date'],
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Channel Performance
    st.subheader("Channel Performance")
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Total Spend vs. Attributed Revenue Bar Chart
        if not daily_marketing_performance_filtered.empty:
            platform_summary = daily_marketing_performance_filtered.groupby('platform').agg(
                total_spend=('total_spend', 'sum'),
                total_attributed_revenue=('total_attributed_revenue', 'sum')
            ).reset_index()
            fig_platform = px.bar(
                platform_summary,
                x='platform',
                y=['total_spend', 'total_attributed_revenue'],
                title="Total Spend vs. Attributed Revenue by Platform",
                labels={'value': 'Amount ($)', 'platform': 'Marketing Platform', 'variable': 'Metric'},
                barmode='group',
                template="plotly_white",
                color_discrete_sequence=['#4B4B4B', '#0072B2']
            )
            st.plotly_chart(fig_platform, use_container_width=True)
    with chart_col4:
        # ROAS by Marketing Platform Bar Chart
        if not daily_marketing_performance_filtered.empty:
            platform_roas_summary = daily_marketing_performance_filtered.groupby('platform').agg(
                total_spend=('total_spend', 'sum'),
                total_attributed_revenue=('total_attributed_revenue', 'sum')
            ).reset_index()
            platform_roas_summary['roas'] = platform_roas_summary.apply(
                lambda x: x['total_attributed_revenue'] / x['total_spend'] if x['total_spend'] > 0 else 0,
                axis=1
            )
            fig_roas = px.bar(
                platform_roas_summary.sort_values('roas', ascending=False),
                x='platform',
                y='roas',
                title="ROAS by Marketing Platform",
                labels={'roas': 'ROAS', 'platform': 'Marketing Platform'},
                text_auto=True,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_roas, use_container_width=True)

    st.markdown("---")

    # Revenue & Spend Analysis
    st.subheader("Revenue & Spend Analysis")
    chart_col5, chart_col6 = st.columns(2)

    with chart_col5:
        # Attributed Revenue by Platform Pie Chart
        if not daily_marketing_performance_filtered.empty:
            revenue_by_platform = daily_marketing_performance_filtered.groupby('platform').agg(
                total_attributed_revenue=('total_attributed_revenue', 'sum')
            ).reset_index()
            fig_pie = px.pie(
                revenue_by_platform,
                names='platform',
                values='total_attributed_revenue',
                title='Attributed Revenue by Platform',
                hole=0.4,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with chart_col6:
        # Daily Spend Heatmap
        if not daily_marketing_performance_filtered.empty:
            spend_by_date_platform = daily_marketing_performance_filtered.pivot_table(
                index='date', 
                columns='platform', 
                values='total_spend', 
                aggfunc='sum'
            ).fillna(0)
            fig_heatmap = px.imshow(
                spend_by_date_platform.T,
                x=spend_by_date_platform.index,
                y=spend_by_date_platform.columns,
                color_continuous_scale='blues',
                title="Daily Spend by Platform",
                template="plotly_white"
            )
            fig_heatmap.update_xaxes(side="top")
            st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # Geographic and Customer-Centric Insights
    st.subheader("Geographic & Customer Insights")
    chart_col7, chart_col8 = st.columns(2)

    with chart_col7:
        # Revenue by State Bar Chart
        if not filtered_df.empty and 'state' in filtered_df.columns:
            revenue_by_state = filtered_df.groupby('state')['total_revenue'].sum().reset_index()
            fig_revenue_state = px.bar(
                revenue_by_state.sort_values('total_revenue', ascending=False),
                x='state',
                y='total_revenue',
                title="Total Revenue by State",
                labels={'total_revenue': 'Total Revenue ($)', 'state': 'State'},
                text_auto=".2s",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_revenue_state, use_container_width=True)
        else:
            st.warning("The 'state' column is not available in the dataset for this chart.")

    with chart_col8:
        # Customer Acquisition Cost (CAC) Bar Chart
        if not filtered_df.empty:
            new_customers_by_platform = filtered_df.groupby('platform')['new_customers'].sum().reset_index()
            spend_by_platform = filtered_df.groupby('platform')['spend'].sum().reset_index()
            cac_df = pd.merge(new_customers_by_platform, spend_by_platform, on='platform')
            cac_df['cac'] = cac_df.apply(
                lambda x: x['spend'] / x['new_customers'] if x['new_customers'] > 0 else 0,
                axis=1
            )
            fig_cac = px.bar(
                cac_df.sort_values('cac', ascending=False),
                x='platform',
                y='cac',
                title="Customer Acquisition Cost (CAC) by Platform",
                labels={'cac': 'CAC ($)', 'platform': 'Marketing Platform'},
                text_auto=".2s",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_cac, use_container_width=True)

    st.markdown("---")

    # Final section: Raw Data Table
    st.subheader("Raw Data")
    if not filtered_df.empty:
        st.dataframe(filtered_df.head(100), use_container_width=True)
    else:
        st.info("No data to display in the raw table.")


else:
    st.error("Data could not be loaded. Please check your CSV files and directory.")