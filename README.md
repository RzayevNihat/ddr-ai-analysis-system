# DDR AI Analysis System 🛢️

**Süni İntellekt ilə Gündəlik Qazma Hesabatlarının Avtomatlaşdırılmış Analizi**

## 📋 Layihə Haqqında

Bu layihə Daily Drilling Reports (DDR) fayllarını avtomatik olaraq oxuyur, başa düşür və analiz edir. Sistem document-level computer vision, domain-specific NLP və LLM əsaslı RAG (Retrieval Augmented Generation) texnologiyalarını birləşdirərək qazma əməliyyatları haqqında dərin insight təqdim edir.

## 🌐 Demo

Canlı demo: [https://ddr-ai-analysis-system.streamlit.app/](https://ddr-ai-analysis-system.streamlit.app/)

## ✨ Əsas Xüsusiyyətlər

### 1. 📄 PDF Parsing & Struktur Analizi
- DDR fayllarının avtomatik strukturlaşdırılması
- Bölmələrin (sections) aşkarlanması
- Cədvəl və rəqəm məlumatlarının çıxarılması
- Quyu adı, operator, dərinlik və s. metadata-nın ekstraksiyası

### 2. 🤖 NLP & Hadisə Təsnifatı
- Qazma fəaliyyətlərinin klassifikasiyası (drilling, tripping, reaming, və s.)
- Anomaliya aşkarlama (stuck pipe, lost circulation, high gas)
- Gündəlik xülasələrin AI ilə yaradılması
- Parametr ekstraksiyası və trend analizi

### 3. 🕸️ Bilik Qrafı (Knowledge Graph)
**Düyümlər (Nodes):**
- Quyular (wellbores)
- Fəaliyyətlər (activities)
- Dərinliklər (depths)
- Formasiyalar (formations)
- Litologiya
- Qazma mayeləri (fluids)
- Anomaliyalar

**Əlaqələr (Edges):**
- Temporal (zaman ardıcıllığı)
- Spatial (məkan əlaqəsi)
- Causal (səbəb-nəticə)

**Sorğu nümunələri:**
```
- "Qaz pikləri > 1.2% olan bütün intervalları göstər"
- "Nüvə nümunələri nə vaxt götürülüb və litologiyası nədir?"
- "2800m dərinlikdə hansı fəaliyyətlər icra olunub?"
```

### 4. 💬 RAG-əsaslı Sual-Cavab Sistemi
- Təbii dildə (Azərbaycanca) suallar
- Vektor bazası ilə ən relevant məlumatların tapılması
- LLM ilə kontekst-aware cavablar
- Mənbə izləmə (source tracking)

### 5. 📊 İnteraktiv Dashboard
- Ümumi statistikalar
- Anomaliya izləmə
- Dərinlik irəliləyişi qrafikləri
- Qaz trend analizi
- Fərdi hesabat baxışı

## 🏗️ Arxitektura

```
ddr-ai-system/
│
├── data/
│   ├── pdfs/                    # İlkin PDF faylları
│   └── processed/
│       ├── processed_ddrs.json  # Emal edilmiş məlumat
│       ├── trends.json          # Trend analizi
│       ├── knowledge_graph.html # KG vizualizasiyası
│       └── embedding_cache/     # Embedding keşi
│
├── chroma_db/                   # Vektor bazası
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Konfiqurasiya
│   ├── pdf_processor.py        # PDF parsing
│   ├── nlp_processor.py        # NLP analizi
│   ├── knowledge_graph.py      # Bilik qrafı
│   ├── llm_service.py          # LLM xidməti (Groq)
│   └── rag_system.py           # RAG sistemi
│
├── app.py                      # Streamlit tətbiqi
├── setup_data.py              # Məlumat emal skripti
├── requirements.txt
├── .env                       # API açarları
└── README.md
```

## 🚀 Quraşdırma

### 1. Repository-ni klonlayın
```bash
git clone <repository-url>
cd ddr-ai-system
```

### 2. Virtual mühit yaradın
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Asılılıqları quraşdırın
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
`.env` faylı yaradın:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
PDF_DATA_PATH=./data/pdfs
PROCESSED_DATA_PATH=./data/processed
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
```

**Groq API Key əldə etmək üçün:** [https://console.groq.com](https://console.groq.com)

### 5. PDF faylları əlavə edin
DDR PDF fayllarını `data/pdfs/` qovluğuna yerləşdirin.

### 6. Məlumatları emal edin
```bash
python setup_data.py
```

Bu skript:
- ✅ PDF-ləri parse edir
- ✅ NLP analizi aparır
- ✅ Bilik qrafı qurur
- ✅ RAG vektor bazası yaradır
- ✅ Trend analizlərini hesablayır

### 7. Tətbiqi işə salın
```bash
streamlit run app.py
```

Brauzerdə açılacaq: `http://localhost:8501`

## 📖 İstifadə

### Dashboard
- **Ümumi İcmal:** Hesabat sayı, quyu sayı, anomaliya statistikaları
- **Son Anomaliyalar:** Real-time anomaliya izləmə
- **Dərinlik İrəliləyişi:** Zamanla qazma proqressi

### Axtarış və Sual-Cavab
Təbii dildə suallar verin:
```
- "Qaz pikləri > 1.2% olan bütün intervalları göstər"
- "15/9-19 B quyusunda hansı litologiyalar müşahidə olunub?"
- "Bütün stuck pipe hadisələrini sadala"
- "3000m dərinlikdə qazma məhlulunun sıxlığı nə qədər olub?"
```

### Trend Analizi
- Dərinlik irəliləyişi trendi
- Qaz göstəriciləri trendi
- Anomaliya zaman qrafiki

### Bilik Qrafı
- İnteraktiv qraf vizualizasiyası
- Sorğu interfeysi:
  - Qaz pikləri
  - Dərinlikdə fəaliyyətlər
  - Litologiya sorğuları

### Fərdi Hesabatlar
- Quyu və tarix üzrə hesabat seçimi
- AI-generated xülasə
- Əməliyyat cədvəli
- Anomaliya detalları

## 🧠 Texnologiyalar

| Komponent | Texnologiya |
|-----------|------------|
| **PDF Parsing** | pdfplumber, regex |
| **NLP** | spaCy, custom keyword-based classification |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB |
| **LLM** | Groq (Llama 3.3 70B) |
| **Knowledge Graph** | NetworkX, Pyvis |
| **Frontend** | Streamlit |
| **Visualization** | Plotly, Matplotlib |

## 🔧 Rate Limiting

Sistem Groq API rate limitleri ilə işləyir:
- **RPM:** 30 requests/minute
- **TPM:** 18,000 tokens/minute

**Rate limiter xüsusiyyətləri:**
- Proaktiv wait mexanizmi
- Exponential backoff retry strategiyası
- Token və request cache
- Real-time statistika

## 📊 Məlumat Strukturu

### processed_ddrs.json
Hər hesabat üçün:
```json
{
  "filename": "15_9-19_B_1997-11-13.pdf",
  "wellbore": "15/9-19 B",
  "period": "1997-11-13 00:00 - 1997-11-14 00:00",
  "operator": "Statoil",
  "depth_md": 2856.0,
  "operations": [...],
  "lithology": [...],
  "gas_readings": [...],
  "detected_anomalies": [...],
  "ai_summary": "..."
}
```

### trends.json
```json
{
  "depth_progress": [...],
  "gas_trends": [...],
  "anomaly_timeline": [...]
}
```

### knowledge_graph.html
Pyvis ilə yaradılmış interaktiv HTML qraf.

## 🎯 Gələcək İnkişaf

- [ ] Multi-wellbore comparative analysis
- [ ] Predictive anomaly detection using ML
- [ ] Real-time PDF upload və processing
- [ ] Export to Excel/PDF reports
- [ ] Multi-language support (EN, RU)
- [ ] Advanced visualization dashboards
- [ ] Historical data trend prediction
- [ ] Integration with drilling databases

## 🐛 Debugging

**Problem:** PDF-lər parse olunmur
```bash
# PDF strukturunu yoxlayın
python -c "import pdfplumber; pdf = pdfplumber.open('data/pdfs/example.pdf'); print(pdf.pages[0].extract_text())"
```

**Problem:** Rate limit xətası
- `.env` faylında GROQ_API_KEY-i yoxlayın
- Rate limiter parametrlərini `llm_service.py`-də tənzimləyin

**Problem:** ChromaDB xətası
```bash
# Vektor bazasını sıfırlayın
rm -rf chroma_db/
python setup_data.py
```

## 📁 Əsas Fayllar

| Fayl | Təsvir |
|------|--------|
| `processed_ddrs.json` | Emal edilmiş DDR məlumatları |
| `trends.json` | Trend analizi nəticələri |
| `knowledge_graph.html` | Bilik qrafının vizualizasiyası |
| `chroma_db/` | Vektor bazası (ChromaDB) |
| `embedding_cache/` | Embedding keşi (performance optimization) |

