"""
app.py — Surface Defect Inspection Dashboard
LozanoLsa · Operational Excellence · ML Portfolio · 2026

Model: Gaussian Naive Bayes — Pipeline (ColumnTransformer + GaussianNB)
Domain: Quality Inspection — Painted Plastic Parts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Surface Defect Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH    = "surface_defect_inspection_data.csv"
RANDOM_STATE = 42
NUM_COLS = [
    "regrind_pct", "resin_temp_c", "cooling_time_s",
    "paint_viscosity", "film_thickness_um", "booth_humidity_pct",
    "pre_paint_storage_hrs", "num_handlings"
]
CAT_COLS = ["container_type", "part_protection"]
TARGET   = "surface_defect"

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="metric-container"] {
        background: #1E2130;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 3px solid #4C9BE8;
    }
</style>
""", unsafe_allow_html=True)

METRIC_EXPLANATIONS = {
    "Accuracy":  "Out of every 100 parts, the model classifies this many correctly.",
    "Precision": "When the model flags a part as defective, this is how often it's actually defective.",
    "Recall":    "Out of all truly defective parts, this is how many the model catches.",
    "F1 Score":  "Balances precision and recall. Closer to 100% = better at both.",
    "AUC-ROC":   "How well the model separates defective from non-defective parts across all thresholds.",
}

ACTION_MAP = {
    "pre_paint_storage_hrs":            "Reduce pre-paint waiting time — implement strict FIFO and buffer zone limits (target < 8 hrs)",
    "num_handlings":                    "Reduce handling steps — redesign internal flow to minimize part contact (target <= 3)",
    "part_protection_unprotected":      "Mandate part protection for all containers — especially metal racks and long-distance moves",
    "container_type_metal_rack":        "Evaluate metal rack lining or foam inserts — direct metal contact causes micro-scratches",
    "container_type_cardboard_pallet":  "Replace cardboard pallets — they introduce dust and moisture absorption",
    "booth_humidity_pct":               "Control paint booth humidity — keep below 60% to prevent adhesion failures",
    "regrind_pct":                      "Limit regrind percentage — above 25% increases surface porosity and paint adhesion issues",
    "resin_temp_c":                     "Monitor resin temperature — maintain within spec (225-245 C) throughout the run",
}


@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return pd.read_csv(
            "https://raw.githubusercontent.com/LozanoLsa/02-Naive-Bayes-Surface-Defect/main/surface_defect_inspection_data.csv"
        )


@st.cache_resource
def train_model(df):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    prep = ColumnTransformer([
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS)
    ])
    model = Pipeline([("prep", prep), ("clf", GaussianNB())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "AUC-ROC":   roc_auc_score(y_test, y_prob),
    }
    return model, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics


def cohens_d(g0, g1):
    n0, n1 = len(g0), len(g1)
    s = np.sqrt(((n0-1)*g0.std()**2+(n1-1)*g1.std()**2)/(n0+n1-2))
    return abs((g1.mean()-g0.mean())/s)


df = load_data()
model, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics = train_model(df)
defect_rate = df[TARGET].mean()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Surface Defect Predictor")
    st.markdown(
        "Gaussian Naive Bayes model trained on 1,500 simulated parts. "
        "Estimates defect probability based on injection, logistics, and paint cabin conditions."
    )
    st.divider()
    st.markdown("### Part Configuration")

    st.markdown("**Internal Logistics**")
    container_type         = st.selectbox("Container Type", ["plastic_box", "metal_rack", "cardboard_pallet"])
    part_protection        = st.selectbox("Part Protection", ["protected", "unprotected"])
    pre_paint_storage_hrs  = st.slider("Pre-Paint Storage (hrs)", 0.0, 24.0, 4.0, 0.5)
    num_handlings          = st.slider("Number of Handlings", 1, 8, 3)

    st.markdown("**Paint Cabin**")
    paint_viscosity    = st.slider("Paint Viscosity", 18.0, 35.0, 25.0, 0.5)
    film_thickness_um  = st.slider("Film Thickness (um)", 20.0, 45.0, 30.0, 0.5)
    booth_humidity_pct = st.slider("Booth Humidity (%)", 25.0, 80.0, 50.0, 1.0)

    st.markdown("**Injection**")
    regrind_pct    = st.slider("Regrind (%)", 0.0, 40.0, 10.0, 0.5)
    resin_temp_c   = st.slider("Resin Temperature (C)", 215.0, 260.0, 235.0, 0.5)
    cooling_time_s = st.slider("Cooling Time (s)", 8.0, 28.0, 18.0, 0.5)

    st.divider()
    st.caption("LozanoLsa · Operational Excellence · ML Portfolio · 2026")


def predict_scenario(ct, pp, psh, nh, pv, ft, bh, rg, rt, cool):
    row = pd.DataFrame([{
        "regrind_pct": rg, "resin_temp_c": rt, "cooling_time_s": cool,
        "paint_viscosity": pv, "film_thickness_um": ft, "booth_humidity_pct": bh,
        "pre_paint_storage_hrs": psh, "num_handlings": nh,
        "container_type": ct, "part_protection": pp
    }])
    prob  = model.predict_proba(row)[0, 1]
    return prob, int(prob >= 0.5)


pred_prob, pred_class = predict_scenario(
    container_type, part_protection, pre_paint_storage_hrs, num_handlings,
    paint_viscosity, film_thickness_um, booth_humidity_pct,
    regrind_pct, resin_temp_c, cooling_time_s
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer", "📈 Model Performance",
    "🎯 Scenario Simulator", "🔍 Risk Drivers", "📋 Action Plan"
])

# ── TAB 1 ────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Parts",  f"{len(df):,}")
    k2.metric("Defective",    f"{df[TARGET].sum():,}")
    k3.metric("Passed",       f"{(df[TARGET]==0).sum():,}")
    k4.metric("Defect Rate",  f"{defect_rate:.1%}")

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        fig_pie = go.Figure(go.Pie(
            labels=["Pass", "Defect"],
            values=[(df[TARGET]==0).sum(), df[TARGET].sum()],
            marker_colors=["#4C9BE8", "#E8574C"],
            hole=0.45, textinfo="percent+label"
        ))
        fig_pie.update_layout(title="Class Distribution", showlegend=False,
                              height=290, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        cat_sel = st.selectbox("Defect rate by:", CAT_COLS)
        rates = df.groupby(cat_sel)[TARGET].mean().reset_index().sort_values(TARGET)
        fig_bar = px.bar(rates, x=TARGET, y=cat_sel, orientation="h",
                         color=TARGET, color_continuous_scale=["#4C9BE8","#E8574C"],
                         labels={TARGET: "Defect Rate", cat_sel: ""},
                         title=f"Defect Rate by {cat_sel.replace('_',' ').title()}")
        fig_bar.update_xaxes(tickformat=".0%")
        fig_bar.update_layout(coloraxis_showscale=False, height=270,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    num_sel = st.selectbox("Numeric feature:", NUM_COLS)
    c3, c4 = st.columns(2)
    with c3:
        fig_h = px.histogram(df, x=num_sel, color=TARGET,
                             color_discrete_map={0:"#4C9BE8",1:"#E8574C"},
                             barmode="overlay", opacity=0.7,
                             title=f"Distribution: {num_sel.replace('_',' ').title()}")
        fig_h.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_h, use_container_width=True)
    with c4:
        fig_bx = px.box(df, x=df[TARGET].map({0:"Pass",1:"Defect"}), y=num_sel,
                        color=df[TARGET].map({0:"Pass",1:"Defect"}),
                        color_discrete_map={"Pass":"#4C9BE8","Defect":"#E8574C"},
                        labels={"x":"","y":num_sel.replace("_"," ").title()},
                        title=f"By Class: {num_sel.replace('_',' ').title()}")
        fig_bx.update_layout(showlegend=False, height=300,
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bx, use_container_width=True)

    df_dum = pd.get_dummies(df, drop_first=True)
    corr_t = df_dum.corr()[[TARGET]].sort_values(TARGET, ascending=False)
    fig_corr = px.imshow(corr_t.T, color_continuous_scale="RdBu_r",
                         zmin=-0.3, zmax=0.3, text_auto=".3f",
                         title="Correlation with Surface Defect", aspect="auto")
    fig_corr.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_corr, use_container_width=True)

# ── TAB 2 ────────────────────────────────────────────────────────────────────────
with tab2:
    st.info(
        f"**How to read this:** Tested on {len(X_test)} unseen parts. "
        f"Classified {metrics['Accuracy']:.0%} correctly. "
        "With a 22% defect rate, Recall and AUC reveal true performance better than Accuracy alone."
    )
    mcols = st.columns(5)
    for col, (name, val) in zip(mcols, metrics.items()):
        col.metric(name, f"{val:.1%}")
        col.caption(METRIC_EXPLANATIONS[name])

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True,
                           x=["Pred: Pass","Pred: Defect"],
                           y=["True: Pass","True: Defect"],
                           color_continuous_scale="Blues", title="Confusion Matrix")
        fig_cm.update_layout(height=360, coloraxis_showscale=False,
                             paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Rows = actual · Columns = predicted · Diagonal = correct")
    with c2:
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Histogram(x=y_prob[y_test==0], name="Actual: Pass",
                                        marker_color="#4C9BE8", opacity=0.7, nbinsx=30))
        fig_prob.add_trace(go.Histogram(x=y_prob[y_test==1], name="Actual: Defect",
                                        marker_color="#E8574C", opacity=0.7, nbinsx=30))
        fig_prob.add_vline(x=0.5, line_dash="dash", line_color="white")
        fig_prob.update_layout(barmode="overlay",
                               title="Predicted Probability by True Class",
                               xaxis_title="Predicted Defect Probability",
                               height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_prob, use_container_width=True)

    rep_df = pd.DataFrame(
        classification_report(y_test, y_pred, target_names=["Pass","Defect"], output_dict=True)
    ).T.round(3)
    st.dataframe(rep_df.style.background_gradient(cmap="Blues",
                                                   subset=["precision","recall","f1-score"]),
                 use_container_width=True)
    st.caption(f"Gaussian Naive Bayes Pipeline | 70/30 stratified split | random_state={RANDOM_STATE}")

# ── TAB 3 ────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Part Defect Probability Simulator")
    left, right = st.columns([1, 2])

    with left:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "Estimated Defect Probability", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E8574C" if pred_class == 1 else "#4C9BE8"},
                "steps": [
                    {"range": [0,  30], "color": "#1a2e1a"},
                    {"range": [30, 60], "color": "#2e2a1a"},
                    {"range": [60, 100], "color": "#2e1a1a"},
                ],
                "threshold": {"line": {"color": "white","width": 3},
                              "thickness": 0.75, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=310, paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(t=60, b=20, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if pred_class == 1:
            st.error("**DEFECT RISK — REJECT**")
        else:
            st.success("**LOW RISK — PASS**")

        st.caption(f"Dataset avg: {defect_rate:.1%}  ·  Your case: {pred_prob:.1%}  ·  Δ {pred_prob - defect_rate:+.1%}")

    with right:
        # Log-likelihood contribution per feature
        clf_obj  = model.named_steps["clf"]
        prep_obj = model.named_steps["prep"]
        X_new_t  = prep_obj.transform(pd.DataFrame([{
            "regrind_pct": regrind_pct, "resin_temp_c": resin_temp_c,
            "cooling_time_s": cooling_time_s, "paint_viscosity": paint_viscosity,
            "film_thickness_um": film_thickness_um, "booth_humidity_pct": booth_humidity_pct,
            "pre_paint_storage_hrs": pre_paint_storage_hrs, "num_handlings": num_handlings,
            "container_type": container_type, "part_protection": part_protection
        }]))
        eps  = 1e-9
        ll0  = -0.5*((X_new_t - clf_obj.theta_[0])**2/(clf_obj.var_[0]+eps)) - 0.5*np.log(2*np.pi*(clf_obj.var_[0]+eps))
        ll1  = -0.5*((X_new_t - clf_obj.theta_[1])**2/(clf_obj.var_[1]+eps)) - 0.5*np.log(2*np.pi*(clf_obj.var_[1]+eps))
        diff = (ll1 - ll0).flatten()
        ohe_feats   = list(prep_obj.named_transformers_["cat"].get_feature_names_out(CAT_COLS))
        all_feats   = NUM_COLS + ohe_feats
        diff_series = pd.Series(diff, index=all_feats)
        top5        = diff_series.abs().nlargest(5).index
        contrib_df  = diff_series[top5].reset_index()
        contrib_df.columns = ["Feature", "Contribution"]

        fig_contrib = go.Figure(go.Bar(
            x=contrib_df["Contribution"],
            y=contrib_df["Feature"].str.replace("_"," ").str.title(),
            orientation="h",
            marker_color=["#E8574C" if v > 0 else "#4C9BE8" for v in contrib_df["Contribution"]]
        ))
        fig_contrib.update_layout(title="What's Driving This Score (Top 5)",
                                  xaxis_title="Log-likelihood diff (class 1 vs 0)",
                                  height=290,
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  margin=dict(t=50,b=40,l=10,r=10))
        st.plotly_chart(fig_contrib, use_container_width=True)

        if pred_prob >= 0.60:
            st.error("**Priority: HIGH** · Hold batch — root cause review before painting")
        elif pred_prob >= 0.30:
            st.warning("**Priority: MEDIUM** · Sample inspection before releasing to paint cabin")
        else:
            st.success("**Priority: LOW** · Standard process — proceed to painting")

    st.divider()
    st.markdown("### Scenario Comparison")
    best_p  = predict_scenario("plastic_box","protected",1.0,2,25.0,30.0,50.0,5.0,235.0,18.0)[0]
    worst_p = predict_scenario("metal_rack","unprotected",18.0,6,21.0,38.0,68.0,28.0,248.0,13.0)[0]
    comp_df = pd.DataFrame([
        {"Scenario":"Best case (controlled process)","P(Defect)":f"{best_p:.1%}",
         "Class":"Pass" if best_p<0.5 else "Defect","Delta vs current":f"{best_p-pred_prob:+.1%}"},
        {"Scenario":"Your current profile","P(Defect)":f"{pred_prob:.1%}",
         "Class":"Pass" if pred_class==0 else "Defect","Delta vs current":"—"},
        {"Scenario":"Worst case (high-risk)","P(Defect)":f"{worst_p:.1%}",
         "Class":"Pass" if worst_p<0.5 else "Defect","Delta vs current":f"{worst_p-pred_prob:+.1%}"},
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ── TAB 4 ────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("What Variables Drive Defect Risk?")

    cd_scores = {c: cohens_d(df[df[TARGET]==0][c], df[df[TARGET]==1][c]) for c in NUM_COLS}
    cd_series = pd.Series(cd_scores).sort_values(ascending=True)
    df_dum2   = pd.get_dummies(df.drop(TARGET, axis=1), drop_first=True)
    mi        = mutual_info_classif(df_dum2, df[TARGET], random_state=RANDOM_STATE)
    mi_s      = pd.Series(mi, index=df_dum2.columns).sort_values(ascending=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cohen's d — Numeric Separation**")
        fig_cd = go.Figure(go.Bar(
            x=cd_series.values,
            y=cd_series.index.str.replace("_"," ").str.title(),
            orientation="h",
            marker_color=["#E8574C" if v > 0.2 else "#4C9BE8" for v in cd_series.values],
            text=cd_series.values.round(3), textposition="outside"
        ))
        fig_cd.add_vline(x=0.2, line_dash="dash", line_color="#FFB347",
                         annotation_text="Small effect")
        fig_cd.update_layout(height=370, xaxis_title="Cohen's d",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cd, use_container_width=True)

    with c2:
        st.markdown("**Mutual Information — All Features**")
        fig_mi = go.Figure(go.Bar(
            x=mi_s.values,
            y=mi_s.index.str.replace("_"," ").str.title(),
            orientation="h",
            marker_color=["#E8574C" if v > mi_s.median() else "#4C9BE8" for v in mi_s.values],
            text=mi_s.values.round(4), textposition="outside"
        ))
        fig_mi.update_layout(height=370, xaxis_title="Mutual Information",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_mi, use_container_width=True)

    st.divider()
    st.markdown("**Digital Pareto — Combined Driver Ranking**")
    cd_norm = pd.Series(cd_scores) / pd.Series(cd_scores).max() * 100
    mi_full = pd.Series(mi, index=df_dum2.columns)
    mi_norm = mi_full / mi_full.max() * 100 if mi_full.max() > 0 else mi_full
    pareto  = pd.concat([cd_norm, mi_norm])
    pareto  = pareto[~pareto.index.duplicated(keep="first")].sort_values(ascending=False)
    cumul   = np.cumsum(pareto.values) / np.cumsum(pareto.values)[-1] * 100

    fig_par = go.Figure()
    fig_par.add_trace(go.Bar(
        x=[f.replace("_"," ").title() for f in pareto.index],
        y=pareto.values,
        marker_color=["#E8574C" if v >= 50 else "#4C9BE8" for v in pareto.values],
        name="Importance (%)"
    ))
    fig_par.add_trace(go.Scatter(
        x=[f.replace("_"," ").title() for f in pareto.index],
        y=cumul, mode="lines+markers",
        line=dict(color="#E8574C", width=2),
        name="Cumulative %", yaxis="y2"
    ))
    fig_par.add_hline(y=80, line_dash="dash", line_color="green",
                      annotation_text="80%", yref="y2")
    fig_par.update_layout(
        title="Digital Pareto — Factors Driving Surface Defect Risk",
        yaxis=dict(title="Normalized Importance (%)"),
        yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right", range=[0,110]),
        legend=dict(orientation="h", y=-0.25),
        height=440,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-40
    )
    st.plotly_chart(fig_par, use_container_width=True)
    st.caption("Cohen's d (numeric) + Mutual Information (all). Directional influence, not causation.")

# ── TAB 5 ────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Operational Action Plan")

    if pred_prob >= 0.60:
        priority_label = "🔴 HIGH"
        horizon = "Before next run"
        action  = "Hold batch — do not release to paint cabin until root cause is identified"
    elif pred_prob >= 0.30:
        priority_label = "🟡 MEDIUM"
        horizon = "Pre-paint inspection"
        action  = "Pull sample for visual pre-inspection — check handling and protection coverage"
    else:
        priority_label = "🟢 LOW"
        horizon = "Standard"
        action  = "Standard process — proceed to paint cabin with normal checklist"

    st.markdown(f"""
| Field | Value |
|---|---|
| **Priority** | {priority_label} |
| **Estimated defect probability** | {pred_prob:.1%} |
| **Suggested action** | {action} |
| **Recommended horizon** | {horizon} |
| **Suggested owner** | Quality Inspector / Process Engineer |
""")

    st.divider()
    st.markdown("### Key Levers for This Part Profile")

    risk_feats = []
    if pre_paint_storage_hrs > 8:
        risk_feats.append("pre_paint_storage_hrs")
    if num_handlings >= 4:
        risk_feats.append("num_handlings")
    if part_protection == "unprotected":
        risk_feats.append("part_protection_unprotected")
    if container_type == "metal_rack":
        risk_feats.append("container_type_metal_rack")
    if container_type == "cardboard_pallet":
        risk_feats.append("container_type_cardboard_pallet")
    if booth_humidity_pct > 60:
        risk_feats.append("booth_humidity_pct")
    if regrind_pct > 25:
        risk_feats.append("regrind_pct")

    if risk_feats:
        for feat in risk_feats[:4]:
            if feat in ACTION_MAP:
                with st.expander(f"▲ {feat.replace('_',' ').title()} — active risk factor"):
                    st.write(ACTION_MAP[feat])
    else:
        st.success("No elevated risk factors detected in the current configuration.")

    st.divider()
    st.caption("_This tool supports quality decisions — it does not replace inspection protocols or process engineering judgment._")
