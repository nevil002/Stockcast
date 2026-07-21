import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "StockCast Dashboard",
    page_icon  = "📦",
    layout     = "wide"
)

st.title("📦 StockCast: End-to-End Demand Forecasting & Inventory Planning")
st.caption("MSc Project — M5 Forecasting | FOODS_1 @ CA_1 | Top 50 Items | 28-Day Horizon")

# ─────────────────────────────────────────
# HELPER — GET WINNING MODEL PREDICTION
# ─────────────────────────────────────────
def get_winning_prediction(selected_model, xgb_data, pro_data, seq_data, tft_data):
    if "XGBoost" in selected_model:
        return xgb_data['xgb_pred']
    elif "Prophet" in selected_model:
        return pro_data['prophet_pred']
    elif "Seq2Seq" in selected_model:
        return seq_data['seq2seq_pred']
    else:
        return tft_data['tft_pred']

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
@st.cache_data
def load_all_data():
    inv_df = pd.read_csv('hybrid_inventory_action_report.csv')
    tft_df = pd.read_csv('tft_output.csv')
    xgb_df = pd.read_csv('xgb_output.csv')
    pro_df = pd.read_csv('prophet_output.csv')
    seq_df = pd.read_csv('seq2seq_output.csv', index_col=0)

    for df in [tft_df, xgb_df, pro_df, seq_df]:
        df['date'] = pd.to_datetime(df['date'])

    try:
        dm_df = pd.read_csv('dm_test_results.csv')
    except FileNotFoundError:
        dm_df = None
        
    try:
        mapping_df = pd.read_csv('product_mapping.csv')
    except FileNotFoundError:
        mapping_df = None

    return inv_df, tft_df, xgb_df, pro_df, seq_df, dm_df, mapping_df

inv_df, tft_df, xgb_df, pro_df, seq_df, dm_df, mapping_df = load_all_data()

# ─────────────────────────────────────────
# PRODUCT MAPPING DICTIONARY
# ─────────────────────────────────────────
if mapping_df is not None:
    mapping_dict = mapping_df.set_index('item_id').apply(
        lambda row: f"{row['Product Name']} ({row['Category']})", axis=1
    ).to_dict()
else:
    mapping_dict = {}

# CHANGED: This function now combines the ID and the mapped name
def format_item_name(item_id):
    friendly_name = mapping_dict.get(item_id)
    if friendly_name:
        return f"{item_id} ➔ {friendly_name}"
    return item_id 

# ─────────────────────────────────────────
# SIDEBAR CONTROLS & EXPORT
# ─────────────────────────────────────────
st.sidebar.header("🎛️ Control Panel")
item_list     = sorted(inv_df['item_id'].unique())

selected_item = st.sidebar.selectbox("🎯 Select a Product:", item_list, format_func=format_item_name)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** StockCast")
st.sidebar.markdown("**Dataset:** M5 Forecasting (Walmart)")
st.sidebar.markdown("**Scope:** FOODS_1 @ CA_1 — Top 50 items")
st.sidebar.markdown("**Horizon:** 28-day ahead forecast")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Data")
csv_data = inv_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Full Inventory Report (CSV)",
    data=csv_data,
    file_name='final_inventory_actions.csv',
    mime='text/csv',
)

# ─────────────────────────────────────────
# ISOLATE ITEM DATA
# ─────────────────────────────────────────
item_info = inv_df[inv_df['item_id'] == selected_item].iloc[0]
tft_data  = tft_df[tft_df['item_id'] == selected_item].sort_values('date').reset_index(drop=True)
xgb_data  = xgb_df[xgb_df['item_id'] == selected_item].sort_values('date').reset_index(drop=True)
pro_data  = pro_df[pro_df['item_id'] == selected_item].sort_values('date').reset_index(drop=True)
seq_data  = seq_df[seq_df['item_id'] == selected_item].sort_values('date').reset_index(drop=True)

if len(tft_data) == 0:
    st.error(f"No prediction data found for {selected_item}. Please select another item.")
    st.stop()

win_pred = get_winning_prediction(item_info['selected_model'], xgb_data, pro_data, seq_data, tft_data)
display_name = format_item_name(selected_item)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Inventory Planning",
    "📈 Model Comparison",
    "⚠️ Residual Analysis",
    "🧪 Statistical Tests (DM)"
])

# ══════════════════════════════════════════
# TAB 1 — INVENTORY PLANNING
# ══════════════════════════════════════════
with tab1:
    st.subheader(f"Supply Chain Status: {display_name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Stock", f"{item_info['current_stock']} units")
    col2.metric("Reorder Point (ROP)", f"{item_info['reorder_point']} units")
    col3.metric("Action Required", item_info['decision'], 
                delta="Urgent" if "REORDER" in item_info['decision'] else "Stable", delta_color="inverse")
    col4.metric("Recommended Order Qty", f"{item_info['suggested_order']} units")

    st.info(f"🧠 **Dynamic Routing:** Forecast powered by **{item_info['selected_model']}** — selected via Diebold-Mariano statistical test.")

    fig_inv = go.Figure()

    fig_inv.add_trace(go.Scatter(
        x = tft_data['date'],
        y = win_pred,
        name = f"📈 Forecasted Demand ({item_info['selected_model']})",
        line = dict(color='royalblue', width=3)
    ))

    fig_inv.add_trace(go.Scatter(
        x = [tft_data['date'].min(), tft_data['date'].max()],
        y = [item_info['reorder_point'], item_info['reorder_point']],
        mode="lines",
        name=f"🔴 Reorder Point (ROP: {item_info['reorder_point']})",
        line=dict(color="red", width=2, dash="dash")
    ))

    fig_inv.add_trace(go.Scatter(
        x = [tft_data['date'].min(), tft_data['date'].max()],
        y = [item_info['current_stock'], item_info['current_stock']],
        mode="lines",
        name=f"🟢 Current Stock ({item_info['current_stock']})",
        line=dict(color="green", width=2, dash="solid")
    ))

    fig_inv.update_layout(
        title = f"28-Day Demand Forecast vs Inventory Thresholds — {display_name}",
        xaxis_title = "Date",
        yaxis_title = "Units",
        height = 420,
        hovermode = "x unified",
        showlegend = True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_inv, use_container_width=True)

    st.markdown("### 📋 Decision Logic")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Current Stock | {item_info['current_stock']} units |
        | Reorder Point | {item_info['reorder_point']} units |
        | Suggested Order | {item_info['suggested_order']} units |
        | Decision | **{item_info['decision']}** |
        """)
    with col_b:
        st.markdown("""
        **How this is calculated:**
        - **ROP** = (7-Day Lead Time Demand) + Statistical Safety Stock
        - **Safety Stock** relies on the standard deviation of the ML forecast to buffer against volatility.
        - **Trigger:** An order is placed immediately if Current Stock ≤ ROP.
        """)

# ══════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════
with tab2:
    st.subheader("Head-to-Head: All 4 Models vs Actual Sales")
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=tft_data['date'], y=tft_data['actual'], name="✅ Actual Sales", line=dict(color='white', width=3, dash='dot')))
    fig_comp.add_trace(go.Scatter(x=tft_data['date'], y=tft_data['tft_pred'], name="TFT", line=dict(color='purple', width=1.5)))
    fig_comp.add_trace(go.Scatter(x=xgb_data['date'], y=xgb_data['xgb_pred'], name="XGBoost", line=dict(color='steelblue', width=1.5)))
    fig_comp.add_trace(go.Scatter(x=pro_data['date'], y=pro_data['prophet_pred'], name="Prophet", line=dict(color='darkorange', width=1.5)))
    fig_comp.add_trace(go.Scatter(x=seq_data['date'], y=seq_data['seq2seq_pred'], name="Seq2Seq", line=dict(color='green', width=1.5)))

    fig_comp.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown(f"### 📊 Item-Level Metrics for {display_name}")
    actual_vals = tft_data['actual'].values

    def safe_mae(actual, pred): return round(mean_absolute_error(actual, pred), 4) if len(pred) > 0 else np.nan
    def safe_rmse(actual, pred): return round(np.sqrt(mean_squared_error(actual, pred)), 4) if len(pred) > 0 else np.nan
    def safe_mase(actual, pred):
        naive = mean_absolute_error(actual[1:], actual[:-1])
        return round(mean_absolute_error(actual, pred) / naive, 4) if naive > 0 else np.nan

    metrics_df = pd.DataFrame({
        'Model': ['TFT', 'XGBoost', 'Prophet', 'Seq2Seq'],
        'MAE': [safe_mae(actual_vals, tft_data['tft_pred'].values), safe_mae(actual_vals, xgb_data['xgb_pred'].values), safe_mae(actual_vals, pro_data['prophet_pred'].values), safe_mae(actual_vals, seq_data['seq2seq_pred'].values)],
        'RMSE': [safe_rmse(actual_vals, tft_data['tft_pred'].values), safe_rmse(actual_vals, xgb_data['xgb_pred'].values), safe_rmse(actual_vals, pro_data['prophet_pred'].values), safe_rmse(actual_vals, seq_data['seq2seq_pred'].values)],
        'MASE': [safe_mase(actual_vals, tft_data['tft_pred'].values), safe_mase(actual_vals, xgb_data['xgb_pred'].values), safe_mase(actual_vals, pro_data['prophet_pred'].values), safe_mase(actual_vals, seq_data['seq2seq_pred'].values)]
    })

    best_mae, best_rmse, best_mase = metrics_df['MAE'].min(), metrics_df['RMSE'].min(), metrics_df['MASE'].min()

    def highlight_best(row):
        styles = [''] * len(row)
        highlight = 'background-color: #d4edda; color: #000000; font-weight: bold;'
        if row['MAE'] == best_mae: styles[1] = highlight
        if row['RMSE'] == best_rmse: styles[2] = highlight
        if row['MASE'] == best_mase: styles[3] = highlight
        return styles

    st.dataframe(metrics_df.style.apply(highlight_best, axis=1), use_container_width=True)

# ══════════════════════════════════════════
# TAB 3 — RESIDUAL ANALYSIS
# ══════════════════════════════════════════
with tab3:
    st.subheader(f"Diagnostic View: Error & Financial Risk for {display_name}")
    
    residual_model = st.selectbox("Select model for residual analysis:", ['XGBoost', 'Prophet', 'Seq2Seq', 'TFT'], index=0)

    pred_map = {'XGBoost': xgb_data['xgb_pred'].values, 'Prophet': pro_data['prophet_pred'].values, 'Seq2Seq': seq_data['seq2seq_pred'].values, 'TFT': tft_data['tft_pred'].values}
    
    residuals_df = pd.DataFrame({'Date': tft_data['date'].values, 'Actual': actual_vals, 'Predicted': pred_map[residual_model]})
    residuals_df['Residual'] = residuals_df['Actual'] - residuals_df['Predicted']
    residuals_df['Type'] = residuals_df['Residual'].apply(lambda r: 'Under-Predicted (Lost Sales)' if r > 0 else 'Over-Predicted (Dead Stock)')

    fig_res = px.bar(residuals_df, x='Date', y='Residual', color='Type',
                     color_discrete_map={'Under-Predicted (Lost Sales)': 'indianred', 'Over-Predicted (Dead Stock)': 'steelblue'})
    fig_res.add_hline(y=0, line_width=2, line_color="white") 
    fig_res.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_res, use_container_width=True)

    avg_price = 3.99 # Mock unit price
    total_under = residuals_df[residuals_df['Residual'] > 0]['Residual'].sum()
    total_over = abs(residuals_df[residuals_df['Residual'] < 0]['Residual'].sum())

    st.markdown("### 📉 Financial & Error Impact Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Error (MAE)", f"{abs(residuals_df['Residual']).mean():.2f} units")
    col2.metric("Days Exact (±0.5)", f"{(abs(residuals_df['Residual']) <= 0.5).sum()} / 28 days")
    col3.metric("Lost Revenue Risk", f"${(total_under * avg_price):.2f}", help="Total under-predicted units × $3.99 avg price")
    col4.metric("Dead Stock Capital", f"${(total_over * avg_price):.2f}", help="Total over-predicted units × $3.99 avg price")

# ══════════════════════════════════════════
# TAB 4 — DIEBOLD-MARIANO TEST RESULTS
# ══════════════════════════════════════════
with tab4:
    st.subheader("Diebold-Mariano Statistical Test Results")
    
    if dm_df is not None:
        if 'TFT Win Rate (%)' not in dm_df.columns:
            dm_df['TFT Win Rate (%)'] = (dm_df['TFT Better'] / 50 * 100).round(1)
            dm_df['Rival Win Rate (%)'] = (dm_df['Rival Better'] / 50 * 100).round(1)
            dm_df['Tie Rate (%)'] = (dm_df['Tie (No Sig. Diff)'] / 50 * 100).round(1)

        st.dataframe(dm_df, use_container_width=True)

        dm_melted = dm_df.melt(id_vars='Rival Model', value_vars=['TFT Better', 'Rival Better', 'Tie (No Sig. Diff)'], var_name='Outcome', value_name='Number of Items')
        fig_dm = px.bar(dm_melted, x='Rival Model', y='Number of Items', color='Outcome', barmode='group',
                        color_discrete_map={'TFT Better': 'steelblue', 'Rival Better': 'tomato', 'Tie (No Sig. Diff)': 'lightgray'}, text_auto=True)
        fig_dm.update_layout(height=420, yaxis_title="Number of Items (out of 50)")
        st.plotly_chart(fig_dm, use_container_width=True)
    else:
        st.warning("dm_test_results.csv not found. Ensure the DM test notebook results are saved in the directory.")