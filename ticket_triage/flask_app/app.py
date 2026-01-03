import streamlit as st
import pandas as pd
import time
import re
import os
import altair as alt
from risk_engine import RiskEngine

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="TriageAI | Risk Protocol",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px !important; font-family: 'Inter', sans-serif; }
    .decision-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    .decision-auto { background-color: #d1e7dd; border-left: 5px solid #0f5132; color: #0f5132; }
    .decision-human { background-color: #f8d7da; border-left: 5px solid #842029; color: #842029; }
    .highlight-risk { background-color: #ffcccc; color: #cc0000; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def highlight_risky_terms(text):
    DANGER_TERMS = ["hacked", "stolen", "data lost", "data gone", "deleted", "account locked", "refund", "crash", "bug", "security", "password", "auth", "login"]
    highlighted = text
    for term in sorted(DANGER_TERMS, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(f'<span class="highlight-risk">{term}</span>', highlighted)
    return highlighted

@st.cache_resource
def load_engine():
    return RiskEngine()

engine = load_engine()

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
    else:
        st.header("🛡️ TriageAI")

    st.markdown("### **Risk Protocol**")
    st.divider()

    st.subheader("⚙️ Tolerance Config")
    risk_mode = st.radio(
        "Sensitivity Level",
        ["Conservative", "Balanced", "Aggressive"],
        index=1
    )
    
    st.divider()

    app_mode = st.radio(
        "WORKSTATION MODE",
        ["Batch Factory", "Sandbox Debugger"],
        index=0
    )
    
    st.markdown("---")
    st.write("")
    
    status_icon = "🟢" if engine.loaded else "🔴"
    st.markdown(f"**Engine Status:** {status_icon} **Online**")
    st.caption(f"Profile: {risk_mode}")

# ==========================================
# MODE 1: BATCH FACTORY
# ==========================================
if app_mode == "Batch Factory":
    st.title("🏭 Auditor Workstation")
    st.markdown("Upload tickets. **Audit** the AI. **Correct** the labels.")
    
    uploaded_file = st.file_uploader("Upload Batch (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            text_col = st.selectbox("Select Ticket Text Column", df.columns)
        with col2:
            st.write("")
            run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        if run_btn:
            with st.status("Analyzing Batch...", expanded=True) as status:
                results = []
                bar = st.progress(0)
                
                for i, row in df.iterrows():
                    text = str(row[text_col])
                    pred = engine.predict(text, mode=risk_mode.lower())
                    
                    # Risk score is already calculated in engine now
                    risk_score = pred['metrics']['risk_score']
                    
                    results.append({
                        "Ticket ID": i,
                        "Text": text,
                        "Decision": pred['decision'],
                        "Issue": pred['signals']['issue'],
                        "Intent": pred['signals']['intent'],
                        "AI_Issue": pred['signals']['issue'],
                        "AI_Intent": pred['signals']['intent'],
                        "Risk Score": risk_score,
                        "Force Safe": False
                    })
                    if i % 10 == 0: bar.progress((i+1)/len(df))
                
                bar.progress(100)
                st.session_state['batch_df'] = pd.DataFrame(results)
                status.update(label="Analysis Complete", state="complete", expanded=False)

    if 'batch_df' in st.session_state:
        data = st.session_state['batch_df']
        
        # --- ANALYTICS ---
        with st.expander("📊 Analytics Dashboard", expanded=True):
            auto_n = len(data[data['Decision'] == 'AUTOMATE'])
            override_n = len(data[data['Decision'] == 'OVERRIDE'])
            total_auto = auto_n + override_n
            total_n = len(data)

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Throughput", total_n)
            kpi2.metric("Automation Rate", f"{(total_auto/total_n):.1%}")
            kpi3.metric("Human Queue", len(data[data['Decision'] == 'HUMAN_REVIEW']), delta="Pending", delta_color="inverse")

        st.divider()

        # --- HITL EDITOR ---
        st.header("🕵️ Correction & Override Desk")
        
        tab_work, tab_export = st.tabs(["📝 Audit Queue", "📤 Export Center"])
        
        with tab_work:
            # 1. THE CORRECT 8 LABELS (Your Taxonomy)
            STANDARD_ISSUES = [
                "PAYMENT_PROBLEM",
                "DATA_LOSS",
                "ACCOUNT_ACCESS",
                "SOFTWARE_BUG",
                "CONNECTIVITY_ISSUE",
                "HARDWARE_FAILURE",
                "DELIVERY_PROBLEM",
                "GENERAL_SUPPORT"
            ]
            
            STANDARD_INTENTS = ["WANTS_INFO", "WANTS_ACTION", "WANTS_SUPPORT"]

            # 2. FILTER
            dataset_issues = data['Issue'].unique().tolist()
            # Combine current predictions with Standard list for filter options
            filter_options = sorted(list(set(dataset_issues + STANDARD_ISSUES)))
            sel_issue = st.multiselect("Filter by Issue", filter_options, default=dataset_issues)
            
            # 3. VIEW PREP
            queue_view = data[
                (data['Decision'] == 'HUMAN_REVIEW') & 
                (data['Issue'].isin(sel_issue))
            ].sort_values(by="Risk Score", ascending=False)
            
            st.caption(f"Reviewing {len(queue_view)} tickets.")

            if not queue_view.empty:
                edited_queue = st.data_editor(
                    queue_view,
                    column_config={
                        "Force Safe": st.column_config.CheckboxColumn("Force Safe?", width="small"),
                        "Issue": st.column_config.SelectboxColumn("Issue", width="medium", options=STANDARD_ISSUES, required=True),
                        "Intent": st.column_config.SelectboxColumn("Intent", width="medium", options=STANDARD_INTENTS, required=True),
                        "Risk Score": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
                        "Text": st.column_config.TextColumn("Ticket Content", width="large", disabled=True)
                    },
                    disabled=["Ticket ID", "Decision", "Risk Score", "Text", "AI_Issue", "AI_Intent"],
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                    key="editor"
                )
                
                if st.button("💾 Save Changes", type="primary"):
                    updates = edited_queue.set_index('Ticket ID').to_dict('index')
                    
                    for tid, row_data in updates.items():
                        master_idx = st.session_state['batch_df'].index[st.session_state['batch_df']['Ticket ID'] == tid][0]
                        
                        # Update Values
                        st.session_state['batch_df'].at[master_idx, 'Issue'] = row_data['Issue']
                        st.session_state['batch_df'].at[master_idx, 'Intent'] = row_data['Intent']
                        st.session_state['batch_df'].at[master_idx, 'Force Safe'] = row_data['Force Safe']
                        
                        # Smart Logic
                        ai_issue = st.session_state['batch_df'].at[master_idx, 'AI_Issue']
                        ai_intent = st.session_state['batch_df'].at[master_idx, 'AI_Intent']
                        
                        is_corrected = (row_data['Issue'] != ai_issue) or (row_data['Intent'] != ai_intent)
                        
                        if is_corrected:
                            st.session_state['batch_df'].at[master_idx, 'Decision'] = 'HUMAN_CLASSIFIED'
                        elif row_data['Force Safe']:
                            st.session_state['batch_df'].at[master_idx, 'Decision'] = 'OVERRIDE'
                            
                    st.toast("Labels Updated & Overrides Applied.", icon="✅")
                    time.sleep(1)
                    st.rerun()
            else:
                st.success("✅ Queue Cleared!")

        with tab_export:
            st.subheader("Data Export")
            
            def make_pie(df, col, title):
                counts = df[col].value_counts().reset_index()
                counts.columns = ['Category', 'Count']
                
                base = alt.Chart(counts).encode(theta=alt.Theta("Count", stack=True))
                pie = base.mark_arc(outerRadius=100).encode(
                    color=alt.Color("Category"),
                    tooltip=["Category", "Count"]
                )
                text = base.mark_text(radius=120).encode(
                    text="Count",
                    order=alt.Order("Count", sort="descending"),
                    color=alt.value("white")
                )
                st.write(f"**{title}**")
                st.altair_chart(pie + text, use_container_width=True)

            c_pie1, c_pie2 = st.columns(2)
            with c_pie1: make_pie(st.session_state['batch_df'], 'Decision', "Final Decision Breakdown")
            with c_pie2: make_pie(st.session_state['batch_df'], 'Issue', "Final Issue Distribution")
            
            st.divider()
            
            df_auto = st.session_state['batch_df'][st.session_state['batch_df']['Decision'].isin(['AUTOMATE', 'OVERRIDE'])]
            df_human = st.session_state['batch_df'][st.session_state['batch_df']['Decision'].isin(['HUMAN_REVIEW', 'HUMAN_CLASSIFIED'])]
            
            c1, c2 = st.columns(2)
            c1.download_button("📥 Download Automation Queue", df_auto.to_csv(index=False), "automation_queue.csv")
            c2.download_button("📥 Download Human/Correction Queue", df_human.to_csv(index=False), "human_training_data.csv")
            
            st.divider()
            
            st.markdown("#### 🤝 Research Contribution")
            st.caption("Optional: Help improve the model by donating your corrected labels.")
            
            if st.button("📧 Contribute Data (Optional)"):
                with st.spinner("Packaging anonymous data..."):
                    time.sleep(1.5)
                    st.success("Thank you! Data package queued for mehtabsandhu0028@gmail.com")

# ==========================================
# MODE 2: SANDBOX
# ==========================================
elif app_mode == "Sandbox Debugger":
    st.header("🛠️ Logic Debugger")
    
    col_in, col_out = st.columns(2)
    with col_in:
        txt = st.text_area("Test Scenario", height=150)
        if st.button("Analyze"):
            st.session_state['debug'] = engine.predict(txt, mode=risk_mode.lower())
            
    with col_out:
        if 'debug' in st.session_state:
            res = st.session_state['debug']
            cls = "decision-auto" if res['decision'] == "AUTOMATE" else "decision-human"
            icon = "✅" if res['decision'] == "AUTOMATE" else "🛑"
            
            st.markdown(f"""
            <div class="decision-box {cls}">
                <h3>{icon} {res['decision']}</h3>
                <p>{res['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(highlight_risky_terms(txt), unsafe_allow_html=True)
            
            # Debug view for weights
            with st.expander("View Risk Calculation"):
                st.json(res['metrics'])