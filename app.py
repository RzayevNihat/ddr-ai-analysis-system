import streamlit as st
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.config import Config
from src.rag_system import RAGSystem
from src.knowledge_graph import KnowledgeGraph
from src.nlp_processor import NLPProcessor
from src.llm_service import LLMService

import warnings
warnings.filterwarnings("ignore")
import streamlit.components.v1 as components

# Səhifə konfiqurasiyası
st.set_page_config(
    page_title="DDR AI Analiz Sistemi",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Xüsusi CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anomaly-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .anomaly-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .anomaly-low {
        color: #fbc02d;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sessiya vəziyyətinin (Session State) işə salınması
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'auto_run_search' not in st.session_state:
    st.session_state.auto_run_search = False

# Emal edilmiş məlumatların yüklənməsi
@st.cache_data
def load_processed_data():
    """Emal edilmiş DDR məlumatlarını yükləyir"""
    data_file = Config.PROCESSED_DATA_PATH / "processed_ddrs.json"
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = [d for d in data if 'error' not in d]
            return data
    return []

@st.cache_data
def load_trends():
    """Trend analizini yükləyir"""
    trends_file = Config.PROCESSED_DATA_PATH / "trends.json"
    if trends_file.exists():
        with open(trends_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_wellbore_name(ddr_data):
    """Quyu adını çıxarır (ehtiyat məntiqi ilə)"""
    wellbore = ddr_data.get('wellbore', '').strip()
    if not wellbore or wellbore == '':
        filename = ddr_data.get('filename', '')
        if filename:
            parts = filename.replace('.pdf', '').split('_')
            if len(parts) >= 2:
                wellbore = f"{parts[0]}/{parts[1]}"
    return wellbore if wellbore else "Naməlum"

# Əsas tətbiq
def main():
    st.markdown('<h1 class="main-header">🛢️ DDR AI Analiz Sistemi</h1>', unsafe_allow_html=True)
    st.markdown("**Süni İntellekt ilə Gündəlik Qazma Hesabatlarının Avtomatlaşdırılmış Analizi**")

    # Yan Menyu (Sidebar)
    with st.sidebar:
        st.image("eilink_03-1.png", width="stretch")
        st.markdown("---")

        page = st.radio(
            "Naviqasiya",
            ["📊 İdarəetmə Paneli", "🔍 Axtarış və Sual-Cavab", "📈 Trend Analizi", "🕸️ Bilik Qrafı", "📋 Hesabatlar"]
        )

        st.markdown("---")
        st.markdown("### Sistem Statusu")

        processed_data = load_processed_data()
        if processed_data:
            st.success(f"✅ {len(processed_data)} hesabat yükləndi")
            wellbores = list(set([get_wellbore_name(d) for d in processed_data]))
            wellbores = [w for w in wellbores if w and w != "Naməlum"]
            st.info(f"📍 {len(wellbores)} quyu")
        else:
            st.warning("⚠️ Məlumat yoxdur. Əvvəlcə setup_data.py faylını işə salın!")

    # Routing
    if page == "📊 İdarəetmə Paneli":
        dashboard_page(processed_data)
    elif page == "🔍 Axtarış və Sual-Cavab":
        search_page(processed_data)
    elif page == "📈 Trend Analizi":
        trends_page()
    elif page == "🕸️ Bilik Qrafı":
        knowledge_graph_page(processed_data)
    elif page == "📋 Hesabatlar":
        reports_page(processed_data)

def dashboard_page(data):
    st.markdown('<h2 class="sub-header">Ümumi İcmal</h2>', unsafe_allow_html=True)

    if not data:
        st.warning("Məlumat yoxdur. PDF-ləri emal etmək üçün setup_data.py faylını işə salın.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cəmi Hesabatlar", len(data))

    with col2:
        wellbores = list(set([get_wellbore_name(d) for d in data]))
        wellbores = [w for w in wellbores if w and w != "Naməlum"]
        st.metric("Quyular", len(wellbores))

    with col3:
        total_anomalies = sum(len(d.get('detected_anomalies', [])) for d in data)
        st.metric("Cəmi Anomaliyalar", total_anomalies)

    with col4:
        operators = list(set([d.get('operator', '').strip() for d in data if d.get('operator', '').strip()]))
        st.metric("Operatorlar", len(operators))

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🚨 Son Anomaliyalar")
        all_anomalies = []
        for d in data:
            for anomaly in d.get('detected_anomalies', []):
                all_anomalies.append({
                    'Quyu': get_wellbore_name(d),
                    'Tarix': d.get('period', '').split()[0] if d.get('period') else '',
                    'Tip': anomaly.get('type', ''),
                    'Ciddilik': anomaly.get('severity', ''),
                    'Təsvir': (anomaly.get('description', '')[:100] + '...') if anomaly.get('description') else ''
                })

        if all_anomalies:
            df_anomalies = pd.DataFrame(all_anomalies[-10:])
            st.dataframe(df_anomalies, width="stretch")
        else:
            st.info("Anomaliya aşkar edilmədi")

    with col2:
        st.markdown("### 📊 Anomaliya Paylanması")
        if all_anomalies:
            anomaly_counts = pd.DataFrame(all_anomalies)['Tip'].value_counts()
            fig = px.pie(values=anomaly_counts.values, names=anomaly_counts.index)
            st.plotly_chart(fig, width="stretch")

    st.markdown("### 📏 Qazma Dərinliyinin İrəliləyişi")
    depth_data = []
    for d in data:
        if d.get('depth_md') and d.get('period'):
            depth_data.append({
                'Tarix': d.get('period', '').split()[0],
                'Dərinlik (MD)': d.get('depth_md'),
                'Quyu': get_wellbore_name(d)
            })

    if depth_data:
        df_depth = pd.DataFrame(depth_data)
        fig = px.line(df_depth, x='Tarix', y='Dərinlik (MD)', color='Quyu', title='Zamanla Qazma Dərinliyi')
        st.plotly_chart(fig, width="stretch")

def search_page(processed_data):
    """Axtarış və Sual-Cavab səhifəsi"""
    st.markdown('<h2 class="sub-header">🔍 Axtarış və Sual-Cavab</h2>', unsafe_allow_html=True)
    st.markdown("Qazma hesabatları haqqında təbii dildə (Azərbaycanca) suallar verin.")

    with st.expander("💡 Nümunə Suallar", expanded=True):
        examples = [
                    "Quyuların adlarını yaz",
        "Qazma gündəlik hesabatlarında anomaliyalar",
        "Quyular üzrə aparılan əməliyyatların ümumi xülasəsi",
        "Konkret quyu və tarix üzrə görülən işlər",
        "Litologiya və qaz oxunuşları",
        "Bütün məhlul itkisi (lost circulation / no returns) hadisələrini göstər",
        "Lost circulation olan faylları göstər",
        "Anomaliya olan tarixləri qeyd et",
        ]

        col1, col2 = st.columns(2)
        for i, ex in enumerate(examples):
            col = col1 if i % 2 == 0 else col2
            with col:
                if st.button(ex, key=f"ex_btn_{i}", width="stretch"):
                    st.session_state.current_question = ex
                    st.session_state.auto_run_search = True
                    st.rerun()

    # Form submit
    with st.form("qa_form", clear_on_submit=False):
        question = st.text_input(
            "Sualınızı daxil edin:",
            value=st.session_state.current_question,
            placeholder="Məsələn: 15/9-F-10 quyusu üçün əməliyyatları ümumiləşdir",
            key="question_input"
        )
        submitted = st.form_submit_button("🔎 Axtar")

    # current_question reset
    if st.session_state.current_question and question == st.session_state.current_question:
        st.session_state.current_question = ""

    auto_run = st.session_state.get("auto_run_search", False)

    if (submitted or auto_run) and question:
        st.session_state.auto_run_search = False
        q_low = question.lower().strip()

        # Deterministic: wellbore list from JSON (RAG-dən asılı deyil)
        if ("quyu" in q_low and "ad" in q_low) or ("wellbore" in q_low and "list" in q_low):
            wellbores = sorted(list(set([get_wellbore_name(d) for d in (processed_data or [])])))
            wellbores = [w for w in wellbores if w and w != "Naməlum"]

            st.markdown("### 💬 Cavab")
            if wellbores:
                st.success("Sistemdə olan quyular:\n" + "\n".join([f"- {w}" for w in wellbores]))
            else:
                st.warning("Məlumatda quyu adı tapılmadı (processed_ddrs.json boş ola bilər).")
            return

        # Normal RAG
        with st.spinner("Axtarılır və cavab hazırlanır..."):
            try:
                result = st.session_state.rag_system.answer_question(question)

                st.markdown("### 💬 Cavab")
                st.success(result.get("answer", ""))

                with st.expander("📚 İstifadə olunan mənbələr"):
                    for i, source in enumerate(result.get("sources", [])):
                        st.markdown(f"**Mənbə {i+1}:**")
                        st.json(source)

                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result.get("answer", ""),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
            except Exception as e:
                st.error(f"Sualın emalı zamanı xəta: {str(e)}")

    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Çat Tarixçəsi")
        for chat in reversed(st.session_state.chat_history[-5:]):
            with st.container():
                st.markdown(f"**S:** {chat['question']}")
                st.markdown(f"**C:** {chat['answer']}")
                st.caption(f"🕐 {chat['timestamp']}")
                st.markdown("---")

def trends_page():
    st.markdown('<h2 class="sub-header">📈 Trend Analizi</h2>', unsafe_allow_html=True)

    trends = load_trends()
    if not trends:
        st.warning("Trend məlumatı yoxdur. Əvvəlcə setup_data.py faylını işə salın.")
        return

    if trends.get('depth_progress'):
        st.markdown("### 📏 Dərinlik İrəliləyişi Trendi")
        df_depth = pd.DataFrame(trends['depth_progress'])
        if not df_depth.empty:
            fig = px.line(df_depth, x='date', y='depth_md', color='wellbore',
                          title='Kumulyativ Dərinlik İrəliləyişi',
                          labels={'date': 'Tarix', 'depth_md': 'Dərinlik (MD)', 'wellbore': 'Quyu'})
            st.plotly_chart(fig, width="stretch")

    if trends.get('gas_trends'):
        st.markdown("### ⛽ Qaz Göstəriciləri Trendi")
        df_gas = pd.DataFrame(trends['gas_trends'])
        if not df_gas.empty and 'percentage' in df_gas.columns:
            df_gas = df_gas.dropna(subset=['percentage'])
            fig = px.scatter(df_gas, x='depth', y='percentage', color='wellbore',
                             title='Dərinliyə görə Qaz Göstəriciləri',
                             labels={'depth': 'Dərinlik (MD)', 'percentage': 'Qaz %', 'wellbore': 'Quyu'})
            fig.add_hline(y=1.2, line_dash="dash", line_color="red", annotation_text="Hədd (1.2%)")
            st.plotly_chart(fig, width="stretch")

    if trends.get('anomaly_timeline'):
        st.markdown("### 🚨 Anomaliya Zaman Qrafiki")
        df_anomaly = pd.DataFrame(trends['anomaly_timeline'])
        if not df_anomaly.empty:
            anomaly_counts = df_anomaly.groupby(['date', 'type']).size().reset_index(name='count')
            fig = px.bar(anomaly_counts, x='date', y='count', color='type',
                         title='Zamanla Anomaliyalar',
                         labels={'date': 'Tarix', 'count': 'Anomaliya Sayı', 'type': 'Anomaliya Tipi'})
            st.plotly_chart(fig, width="stretch")

def knowledge_graph_page(data):
    st.markdown('<h2 class="sub-header">🕸️ Bilik Qrafı</h2>', unsafe_allow_html=True)
    st.markdown("Qazma fəaliyyətləri, dərinliklər, formasiyalar və anomaliyalar arasındakı əlaqələri araşdırın.")

    kg = KnowledgeGraph()

    if data:
        with st.spinner("Bilik qrafı qurulur..."):
            for ddr in data[:20]:
                kg.build_from_ddr(ddr)

        stats = kg.get_statistics()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cəmi Düyümlər", stats['total_nodes'])
        with col2:
            st.metric("Cəmi Əlaqələr", stats['total_edges'])
        with col3:
            st.metric("Anomaliyalar", stats['anomalies'])

        st.markdown("### 🔍 Bilik Qrafını Sorğula")
        query_type = st.selectbox("Sorğu Növünü Seçin:", ["Qaz Piklər", "Dərinlikdə Fəaliyyətlər", "Dərinlikdə Litologiya"])

        if query_type == "Qaz Piklər":
            threshold = st.slider("Qaz Həddi (%)", 0.5, 5.0, 1.2, 0.1)
            if st.button("Sorğunu İcra Et"):
                results = kg.query_gas_peaks(threshold)
                if results:
                    st.dataframe(pd.DataFrame(results), width="stretch")
                else:
                    st.info("Nəticə tapılmadı")

        elif query_type == "Dərinlikdə Fəaliyyətlər":
            wellbore = st.text_input("Quyu:", value="15/9-19 B")
            depth = st.number_input("Dərinlik (MD):", value=2800.0)
            tolerance = st.number_input("Tolerans (m):", value=10.0)
            if st.button("Sorğunu İcra Et"):
                results = kg.query_activities_at_depth(wellbore, depth, tolerance)
                if results:
                    st.dataframe(pd.DataFrame(results), width="stretch")
                else:
                    st.info("Bu dərinlikdə fəaliyyət tapılmadı")

        if st.button("Qraf Vizualizasiyasını Yarat"):
            with st.spinner("Vizualizasiya yaradılır..."):
                kg.visualize(output_file=str(Config.PROCESSED_DATA_PATH / "kg_viz.html"))
                st.success("Vizualizasiya data/processed/kg_viz.html ünvanında yadda saxlanıldı")
                st.markdown("[Vizualizasiyanı Aç](../data/processed/kg_viz.html)")

    kg_html_path = Config.PROCESSED_DATA_PATH / "knowledge_graph.html"
    if kg_html_path.exists():
        with open(kg_html_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        st.markdown("### 🕸️ Bilik Qrafı Vizualizasiyası")
        components.html(html_data, height=600, scrolling=True)
    else:
        st.warning("Vizualizasiya faylı tapılmadı. Zəhmət olmasa əvvəlcə datanı emal edin.")

def reports_page(data):
    st.markdown('<h2 class="sub-header">📋 Fərdi Hesabatlar</h2>', unsafe_allow_html=True)

    if not data:
        st.warning("Hesabat mövcud deyil")
        return

    wellbores = sorted(list(set([get_wellbore_name(d) for d in data])))
    wellbores = [w for w in wellbores if w and w != "Naməlum"]

    if not wellbores:
        st.error("Məlumatlarda quyu məlumatı tapılmadı. PDF parsing-i yoxlayın.")
        st.info("İpucu: Quyu adlarının PDF-lərdən düzgün çıxarıldığına əmin olun.")
        return

    selected_wellbore = st.selectbox("Quyu Seçin:", wellbores)
    wellbore_reports = [d for d in data if get_wellbore_name(d) == selected_wellbore]
    wellbore_reports = sorted(wellbore_reports, key=lambda x: x.get('period', ''))

    if not wellbore_reports:
        st.warning(f"{selected_wellbore} quyusu üçün hesabat tapılmadı")
        return

    report_options = [f"{r.get('period', 'Naməlum')} - {r.get('filename', '')}" for r in wellbore_reports]
    selected_idx = st.selectbox("Hesabat Seçin:", range(len(report_options)), format_func=lambda x: report_options[x])

    if selected_idx is not None:
        report = wellbore_reports[selected_idx]
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📄 Hesabat Detalları")
            st.markdown(f"**Quyu:** {get_wellbore_name(report)}")
            st.markdown(f"**Dövr:** {report.get('period', 'N/A')}")
            st.markdown(f"**Operator:** {report.get('operator', 'N/A')}")
            st.markdown(f"**Qazma Qurğusu:** {report.get('rig_name', '').split('Depth')[0]}")

            depth = report.get('depth_md')
            st.markdown(f"**Dərinlik (MD):** {f'{depth}m' if depth else 'N/A'}")

            hole_size = report.get('hole_size')
            hole_size_text = f'{hole_size}"' if hole_size else 'N/A'
            st.markdown(f"**Quyu Diametri:** {hole_size_text}")

            if report.get('ai_summary'):
                with st.expander("🤖 AI Xülasə", expanded=True):
                    st.write(report['ai_summary'])

        with col1:
            st.markdown("### 📝 Fəaliyyətlərin İcmalı")
            ai_summary = report.get('summary', {}).get('activities_24h')
            if ai_summary and ai_summary != "N/A":
                st.write(ai_summary)
            else:
                st.info("AI xülasəsi tapılmadı. Orijinal mətn:")
                st.write(report.get('summary_text', 'Məlumat yoxdur'))

        with col2:
            st.markdown("### 🚨 Anomaliyalar")
            anomalies = report.get('detected_anomalies', [])
            if anomalies:
                for anomaly in anomalies:
                    severity = anomaly.get('severity', 'low')
                    color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
                    st.markdown(f"{color} **{anomaly.get('type', 'Unknown').replace('_', ' ').title()}**")
                    st.caption((anomaly.get('description', '') or '')[:150])
                    st.markdown("---")
            else:
                st.success("✅ Anomaliya aşkar edilmədi")

        if report.get('operations'):
            st.markdown("### ⚙️ Əməliyyatlar")
            ops_data = []
            for op in report['operations']:
                ops_data.append({
                    'Başlanğıc': op.get('start_time', ''),
                    'Bitmə': op.get('end_time', ''),
                    'Dərinlik': op.get('depth', ''),
                    'Fəaliyyət': op.get('activity', ''),
                    'Vəziyyət': op.get('state', ''),
                    'Qeyd': (op.get('remark', '') or '')[:100]
                })
            if ops_data:
                st.dataframe(pd.DataFrame(ops_data), width="stretch")

if __name__ == "__main__":
    main()
