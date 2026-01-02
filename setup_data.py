"""
Data processing script - Rate Limit Safe with Checkpoints
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
import logging
from collections import Counter
from datetime import datetime

from src.config import Config
from src.pdf_processor import DDRParser
from src.nlp_processor import NLPProcessor
from src.knowledge_graph import KnowledgeGraph
from src.rag_system import RAGSystem

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== KONFIQURASIYA =====
CHECKPOINT_INTERVAL = 50  # Hər 50 fayldan bir checkpoint
ENABLE_CHECKPOINTS = True  # Checkpoint sistemi
# =========================

def load_checkpoint():
    """Checkpoint-dən davam et"""
    checkpoint_file = Config.PROCESSED_DATA_PATH / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_checkpoint(processed_count, processed_data):
    """Checkpoint yarat"""
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'processed_count': processed_count,
        'processed_data': processed_data
    }
    checkpoint_file = Config.PROCESSED_DATA_PATH / "checkpoint.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Checkpoint saxlandı: {processed_count} fayl")

def validate_wellbore_data(data_list):
    """Quyu məlumatlarının keyfiyyətini yoxlayır"""
    logger.info("\n" + "="*60)
    logger.info("QUYU MƏLUMATLARININ VALİDASİYASI")
    logger.info("="*60)
    
    wellbores = []
    missing_wellbore = 0
    
    for d in data_list:
        wellbore = d.get('wellbore', '').strip()
        if wellbore and wellbore != '' and wellbore != 'Unknown':
            wellbores.append(wellbore)
        else:
            missing_wellbore += 1
    
    wellbore_counts = Counter(wellbores)
    
    logger.info(f"Cəmi Hesabatlar: {len(data_list)}")
    logger.info(f"Quyu adı tapılan: {len(wellbores)}")
    logger.info(f"Quyu adı tapılmayan: {missing_wellbore}")
    logger.info(f"Unikal Quyular: {len(wellbore_counts)}")
    
    logger.info("\nQuyu paylanması:")
    for wb, count in wellbore_counts.most_common(20):
        logger.info(f"  {wb}: {count} hesabat")
    
    return wellbore_counts

def main():
    logger.info("="*60)
    logger.info("DDR PROSESİ BAŞLAYIR (Rate Limit Safe Mode)")
    logger.info("="*60)
    
    # Checkpoint yoxla
    checkpoint = load_checkpoint() if ENABLE_CHECKPOINTS else None
    start_index = 0
    processed_data = []
    
    if checkpoint:
        logger.info(f"📂 Checkpoint tapıldı: {checkpoint['processed_count']} fayl artıq emal olunub")
        response = input("Checkpoint-dən davam etmək istəyirsiniz? (y/n): ")
        if response.lower() == 'y':
            start_index = checkpoint['processed_count']
            processed_data = checkpoint['processed_data']
            logger.info(f"✅ {start_index} fayldan davam edilir")
            logger.info("⏳ Rate limit window təmizlənməsi üçün 90 saniyə gözləyir...")
            time.sleep(90)
            logger.info("✅ Davam edilir")
    # --- STEP 1: PARSING ---
    logger.info("\n[1/5] PDF Faylları Oxunur (Parsing)...")
    parser = DDRParser()
    pdf_files = list(Config.PDF_DATA_PATH.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"Fayl tapılmadı: {Config.PDF_DATA_PATH}")
        return
    
    logger.info(f"{len(pdf_files)} PDF faylı tapıldı")
    
    # Parse (yalnız yeni fayllar)
    all_ddr_data = []
    
    if start_index > 0:
        # Əvvəlki parse məlumatını yüklə
        parsed_file = Config.PROCESSED_DATA_PATH / "parsed_ddrs.json"
        if parsed_file.exists():
            with open(parsed_file, 'r', encoding='utf-8') as f:
                all_ddr_data = json.load(f)
            logger.info(f"✅ {len(all_ddr_data)} əvvəlki parse yükləndi")
    
    # Yeni parsing
    files_to_parse = pdf_files[len(all_ddr_data):]
    if files_to_parse:
        for pdf_path in tqdm(files_to_parse, desc="Parsing PDFs"):
            try:
                ddr_data = parser.parse_pdf(pdf_path)
                all_ddr_data.append(ddr_data)
            except Exception as e:
                logger.error(f"Parsing xətası {pdf_path}: {e}")
                all_ddr_data.append({
                    'filename': pdf_path.name,
                    'error': str(e)
                })
        
        # Saxla
        parsed_file_path = Config.PROCESSED_DATA_PATH / "parsed_ddrs.json"
        with open(parsed_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_ddr_data, f, indent=2, ensure_ascii=False)
    
    validate_wellbore_data(all_ddr_data)
    
# --- STEP 2: NLP PROCESSING ---
    logger.info(f"\n{'='*60}")
    logger.info(f"[2/5] NLP Processing (Rate Limit Safe)")
    logger.info(f"Başlanğıc: {start_index}/{len(all_ddr_data)}")
    logger.info(f"Qalan: {len(all_ddr_data) - start_index} fayl")
    logger.info(f"⏱️  Təxmini vaxt: {(len(all_ddr_data) - start_index) * 15 / 60:.1f} dəqiqə")
    logger.info(f"ℹ️  Rate Limit: 28 req/min, 17k tokens/min (auto-managed)")
    logger.info(f"{'='*60}\n")

    nlp_processor = NLPProcessor()
    failed_files = []

    # Progress bar
    with tqdm(total=len(all_ddr_data), initial=start_index, desc="NLP Processing") as pbar:
        for i in range(start_index, len(all_ddr_data)):
            ddr = all_ddr_data[i]
            
            if 'error' not in ddr:
                try:
                    # Process
                    params = nlp_processor.extract_parameters(ddr)
                    summary = nlp_processor.create_daily_summary(ddr)
                    events = nlp_processor.classify_events(ddr)
                    anomalies = nlp_processor.detect_anomalies(ddr)
                    
                    ddr['extracted_params'] = params
                    ddr['ai_summary'] = summary
                    ddr['classified_events'] = events
                    ddr['detected_anomalies'] = anomalies
                    
                    processed_data.append(ddr)
                    
                    # Update progress
                    pbar.set_postfix({
                        'Uğurlu': len(processed_data),
                        'Xəta': len(failed_files)
                    })
                    
                    # Checkpoint
                    if ENABLE_CHECKPOINTS and (i + 1) % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(i + 1, processed_data)
                        # İntermediate save
                        temp_file = Config.PROCESSED_DATA_PATH / "processed_ddrs_temp.json"
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(processed_data, f, indent=2, ensure_ascii=False)
                    
                except Exception as e:
                    logger.error(f"❌ Xəta: {ddr.get('filename')}: {str(e)}")
                    failed_files.append({
                        'filename': ddr.get('filename'),
                        'error': str(e)
                    })
            else:
                failed_files.append({
                    'filename': ddr.get('filename'),
                    'error': ddr.get('error')
                })
            
            pbar.update(1)
    
    # Rate limit statistikası
    stats = nlp_processor.llm.get_rate_limit_stats()
    logger.info(f"\n{'='*60}")
    logger.info("RATE LIMIT STATİSTİKA")
    logger.info(f"{'='*60}")
    logger.info(f"Ümumi request: {stats['total_requests']}")
    logger.info(f"Ümumi token: {stats['total_tokens']:,}")
    logger.info(f"Rate limit hit: {stats['rate_limit_hits']}")
    logger.info(f"Ümumi gözləmə: {stats['total_wait_time']:.1f}s ({stats['total_wait_time']/60:.1f} dəqiqə)")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EMAL YEKUNU")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Uğurlu: {len(processed_data)}/{len(all_ddr_data)}")
    logger.info(f"❌ Xətalı: {len(failed_files)}/{len(all_ddr_data)}")

    if failed_files:
        logger.warning(f"\nXəta verən fayllar:")
        for f in failed_files[:20]:
            logger.warning(f" - {f['filename']}: {f.get('error', 'Unknown error')[:100]}")
        
        # Save failed files list
        with open(Config.PROCESSED_DATA_PATH / "failed_files.json", 'w', encoding='utf-8') as f:
            json.dump(failed_files, f, indent=2, ensure_ascii=False)

    # Save final
    output_file = Config.PROCESSED_DATA_PATH / "processed_ddrs.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Yekun məlumat saxlanıldı: {output_file}")
    
    # Checkpoint-i sil
    if ENABLE_CHECKPOINTS:
        checkpoint_file = Config.PROCESSED_DATA_PATH / "checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("✅ Checkpoint silindi (proses tamamlandı)")
    
    # --- STEP 3: KNOWLEDGE GRAPH ---
    logger.info("\n[3/5] Bilik Qrafı qurulur...")
    kg = KnowledgeGraph()
    for ddr in tqdm(processed_data, desc="Building KG"):
        try:
            kg.build_from_ddr(ddr)
        except Exception as e:
            logger.debug(f"KG xətası: {e}")
    
    kg_stats = kg.get_statistics()
    logger.info(f"Qraf statistikası: {kg_stats}")
    kg.visualize(output_file=str(Config.PROCESSED_DATA_PATH / "knowledge_graph.html"))
    
    # --- STEP 4: RAG SYSTEM ---
    logger.info("\n[4/5] RAG Vektor Bazası qurulur...")
    try:
        rag = RAGSystem()
        rag.add_documents(processed_data)
        rag_stats = rag.get_statistics()
        logger.info(f"RAG statistikası: {rag_stats}")
    except Exception as e:
        logger.error(f"RAG xətası: {e}")
    
    # --- STEP 5: TREND ANALYSIS ---
    logger.info("\n[5/5] Trend Analizi aparılır...")
    try:
        trends = nlp_processor.analyze_trends(processed_data)
        with open(Config.PROCESSED_DATA_PATH / "trends.json", 'w', encoding='utf-8') as f:
            json.dump(trends, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Trend xətası: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("PROSES TAMAMLANDI!")
    logger.info("="*60)
    logger.info("🚀 Tətbiqi işə salın: streamlit run app.py")

if __name__ == "__main__":
    main()