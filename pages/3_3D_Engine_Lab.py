from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config.config import FD001_ACTIVE_SENSORS, DATASETS, SENSOR_INFO
from src.data_loader import load_dataset

st.set_page_config(page_title="3D Engine Lab • Turbofan RUL",page_icon="🧊",layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#f6f9ff,#fff 48%,#f7f5ff)}
.hero{padding:25px 28px;border-radius:22px;background:linear-gradient(120deg,#172554,#2563eb 52%,#7c3aed);color:white;margin-bottom:18px}
.card{background:white;border:1px solid #e2e8f0;border-radius:18px;padding:18px}
.section{margin:24px 0 10px;font-weight:800;font-size:1.15rem;border-left:5px solid #6366f1;padding-left:10px}
</style>
""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🧊 3D Engine Laboratory</h1><p>Explore turbofan degradation trajectories through PCA sensor space and cycle-by-cycle telemetry.</p></div>',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(name): return load_dataset(name)

with st.sidebar:
    subset=st.selectbox("Dataset",list(DATASETS.keys()))
    _,test,_=load(subset)
    engine=st.selectbox("Engine",sorted(test.unit_number.unique()))

eng=test[test.unit_number==engine].sort_values("time_cycles").copy()
X=eng[FD001_ACTIVE_SENSORS].replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0)
pca=PCA(n_components=3,random_state=42)
pcs=pca.fit_transform(X)
eng["PC1"],eng["PC2"],eng["PC3"]=pcs[:,0],pcs[:,1],pcs[:,2]

a,b,c,d=st.columns(4)
a.metric("Engine",engine)
b.metric("Observed cycles",len(eng))
c.metric("PC1 variance",f"{pca.explained_variance_ratio_[0]:.1%}")
d.metric("3D variance",f"{pca.explained_variance_ratio_.sum():.1%}")

st.markdown('<div class="section">3D sensor trajectory</div>',unsafe_allow_html=True)
fig=px.scatter_3d(eng,x="PC1",y="PC2",z="PC3",color="time_cycles",color_continuous_scale="Turbo",hover_data=["time_cycles"]+FD001_ACTIVE_SENSORS[:4],title=f"Engine {engine} — PCA degradation trajectory")
fig.update_traces(marker=dict(size=5))
fig.update_layout(height=720,margin=dict(l=0,r=0,t=50,b=0))
st.plotly_chart(fig,use_container_width=True)

st.markdown('<div class="section">3D physical sensor view</div>',unsafe_allow_html=True)
z=st.selectbox("Z-axis sensor",FD001_ACTIVE_SENSORS,index=2,format_func=lambda x:f"{x} — {SENSOR_INFO[x]['desc']}")
y=st.selectbox("Y-axis sensor",FD001_ACTIVE_SENSORS,index=6,format_func=lambda x:f"{x} — {SENSOR_INFO[x]['desc']}")
fig2=go.Figure(go.Scatter3d(x=eng.time_cycles,y=eng[y],z=eng[z],mode="lines+markers",marker=dict(size=3,color=eng.time_cycles,colorscale="Turbo",showscale=True),line=dict(width=5)))
fig2.update_layout(scene=dict(xaxis_title="Cycle",yaxis_title=y,zaxis_title=z),height=650,template="plotly_white",margin=dict(l=0,r=0,t=30,b=0))
st.plotly_chart(fig2,use_container_width=True)
