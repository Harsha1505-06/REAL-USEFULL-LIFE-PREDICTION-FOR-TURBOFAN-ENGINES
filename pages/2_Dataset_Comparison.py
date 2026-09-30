from pathlib import Path
import io
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from config.config import ALL_COLUMNS, DATASETS, FD001_ACTIVE_SENSORS, SENSOR_INFO, MODELS_DIR, WINDOW_SIZE
from src.data_loader import load_dataset
from src.features import create_tabular_features, create_test_last_sequence_windows
from src.models import TurbofanLSTM, Turbofan1DCNN

st.set_page_config(page_title="Dataset Comparison • Turbofan RUL",page_icon="📊",layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#f6f9ff,#fff 48%,#f7f5ff)}
.hero{padding:25px 28px;border-radius:22px;background:linear-gradient(120deg,#172554,#2563eb 52%,#7c3aed);color:white;margin-bottom:18px}
.card{background:white;border:1px solid #e2e8f0;border-radius:18px;padding:18px;box-shadow:0 8px 25px #0f172a10}
.section{margin:24px 0 10px;font-weight:800;font-size:1.15rem;border-left:5px solid #6366f1;padding-left:10px}
.info{background:#eff6ff;border:1px solid #bfdbfe;border-radius:14px;padding:13px 16px;color:#1e3a8a}
</style>
""",unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>📊 Dataset Comparison Lab</h1><p>Upload your turbofan telemetry and compare it directly with NASA C-MAPSS reference data.</p></div>',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def reference(name):
    return load_dataset(name)

@st.cache_resource(show_spinner=False)
def models():
    out={}
    for name,file in [("Linear Regression","linear_regression.joblib"),("Random Forest","random_forest.joblib"),("XGBoost","xgboost.joblib")]:
        p=MODELS_DIR/file
        if p.exists(): out[name]=joblib.load(p)
    scaler_path=MODELS_DIR/"minmax_scaler.joblib"
    scaler=joblib.load(scaler_path) if scaler_path.exists() else None
    for name,cls,file in [("LSTM",TurbofanLSTM,"lstm_model.pt"),("1D-CNN",Turbofan1DCNN,"1dcnn_model.pt")]:
        p=MODELS_DIR/file
        if p.exists():
            try:
                m=cls(input_dim=len(FD001_ACTIVE_SENSORS))
                m.load_state_dict(torch.load(p,map_location="cpu"))
                m.eval(); out[name]=m
            except Exception: pass
    return out,scaler

def parse(upload):
    raw=upload.getvalue()
    if upload.name.lower().endswith(".csv"):
        df=pd.read_csv(io.BytesIO(raw))
    else:
        df=pd.read_csv(io.BytesIO(raw),sep=r"\s+",header=None)
        if df.shape[1]==26: df.columns=ALL_COLUMNS
    df=df.dropna(axis=1,how="all")
    aliases={"unit":"unit_number","engine":"unit_number","engine_id":"unit_number","cycle":"time_cycles"}
    df.rename(columns={c:aliases.get(str(c).strip().lower(),c) for c in df.columns},inplace=True)
    return df

def stat(df):
    return {
        "Rows":len(df),
        "Columns":len(df.columns),
        "Engines":int(df.unit_number.nunique()) if "unit_number" in df else 0,
        "Max cycle":int(df.time_cycles.max()) if "time_cycles" in df else 0,
        "Mean engine life":round(float(df.groupby("unit_number").time_cycles.max().mean()),1) if "unit_number" in df and "time_cycles" in df else 0,
        "Missing cells":int(df.isna().sum().sum())
    }

def infer(df):
    mdl,scaler=models()
    if scaler is None or not set(ALL_COLUMNS).issubset(df.columns): return None
    try:
        work=df.copy().sort_values(["unit_number","time_cycles"])
        work[FD001_ACTIVE_SENSORS]=scaler.transform(work[FD001_ACTIVE_SENSORS])
        tab=create_tabular_features(work,FD001_ACTIVE_SENSORS,WINDOW_SIZE)
        cols=FD001_ACTIVE_SENSORS+[f"{c}_roll_mean_{WINDOW_SIZE}" for c in FD001_ACTIVE_SENSORS]+[f"{c}_roll_std_{WINDOW_SIZE}" for c in FD001_ACTIVE_SENSORS]
        last=tab.groupby("unit_number").last().reset_index()
        result=pd.DataFrame({"unit_number":last.unit_number.values})
        for name in ["Linear Regression","Random Forest","XGBoost"]:
            if name in mdl: result[name]=np.maximum(0,mdl[name].predict(last[cols]))
        seq,_=create_test_last_sequence_windows(work,np.zeros(work.unit_number.nunique()),FD001_ACTIVE_SENSORS,WINDOW_SIZE)
        x=torch.tensor(seq,dtype=torch.float32)
        with torch.no_grad():
            if "LSTM" in mdl: result["LSTM"]=np.maximum(0,mdl["LSTM"](x).numpy())
            if "1D-CNN" in mdl: result["1D-CNN"]=np.maximum(0,mdl["1D-CNN"](x).numpy())
        return result
    except Exception:
        return None

with st.sidebar:
    st.markdown("### Reference")
    subset=st.selectbox("C-MAPSS dataset",list(DATASETS.keys()))
    st.caption(f"{DATASETS[subset]['conditions']} operating condition(s) • {DATASETS[subset]['faults']}")

train,test,rul=reference(subset)
uploaded=st.file_uploader("Upload TXT or CSV dataset",type=["txt","csv"])

if not uploaded:
    st.markdown('<div class="info"><b>How to use:</b> upload a 26-column NASA C-MAPSS telemetry file. Choose a reference dataset from the sidebar. The app will compare structure, engine life and sensor distributions, then attempt multi-model RUL inference.</div>',unsafe_allow_html=True)
    st.stop()

try:
    user=parse(uploaded)
    missing=[c for c in ALL_COLUMNS if c not in user.columns]
    if missing:
        st.error("This file is not C-MAPSS compatible. Missing columns: "+", ".join(missing))
        st.dataframe(user.head(10),use_container_width=True)
        st.stop()

    ref_type=st.radio("Reference set",["Test","Train"],horizontal=True)
    ref=test if ref_type=="Test" else train
    us,rs=stat(user),stat(ref)
    st.success(f"Validated **{uploaded.name}** — {len(user):,} rows and 26 telemetry columns.")

    cols=st.columns(6)
    for i,k in enumerate(us):
        cols[i].metric(k,us[k],f"reference {rs[k]}")

    st.markdown('<div class="section">Fleet structure comparison</div>',unsafe_allow_html=True)
    comparison=pd.DataFrame({"Metric":list(us.keys()),"Uploaded":[us[k] for k in us],"Reference":[rs[k] for k in us]})
    comparison["Difference"]=comparison["Uploaded"]-comparison["Reference"]
    st.dataframe(comparison,use_container_width=True,hide_index=True)

    st.markdown('<div class="section">Sensor distribution comparison</div>',unsafe_allow_html=True)
    sensor=st.selectbox("Sensor",FD001_ACTIVE_SENSORS,format_func=lambda x:f"{x} — {SENSOR_INFO[x]['desc']}")
    hist=pd.DataFrame({"Value":pd.concat([user[sensor],ref[sensor]],ignore_index=True),"Dataset":["Uploaded"]*len(user)+["Reference"]*len(ref)})
    st.plotly_chart(px.histogram(hist,x="Value",color="Dataset",barmode="overlay",nbins=50,opacity=.65,title=f"{sensor} distribution",template="plotly_white"),use_container_width=True)

    means=pd.DataFrame({"Sensor":FD001_ACTIVE_SENSORS,"Uploaded":user[FD001_ACTIVE_SENSORS].mean().values,"Reference":ref[FD001_ACTIVE_SENSORS].mean().values}).melt("Sensor",var_name="Dataset",value_name="Mean")
    st.plotly_chart(px.bar(means,x="Sensor",y="Mean",color="Dataset",barmode="group",title="Mean active-sensor comparison",template="plotly_white"),use_container_width=True)

    st.markdown('<div class="section">Uploaded dataset RUL inference</div>',unsafe_allow_html=True)
    pred=infer(user)
    if pred is not None and len(pred):
        st.success(f"Predicted RUL for {len(pred):,} uploaded engines using the available pretrained models.")
        st.dataframe(pred.round(2),use_container_width=True,hide_index=True)
        melt=pred.melt("unit_number",var_name="Model",value_name="Predicted RUL")
        st.plotly_chart(px.box(melt,x="Model",y="Predicted RUL",color="Model",title="Predicted RUL distribution",template="plotly_white"),use_container_width=True)
        st.download_button("⬇️ Download predictions",pred.to_csv(index=False),"uploaded_dataset_predictions.csv","text/csv")
    else:
        st.warning("The dataset comparison completed, but pretrained inference was not possible for this file.")

except Exception as exc:
    st.error(f"Unable to process the uploaded dataset: {exc}")
