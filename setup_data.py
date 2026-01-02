"""
DDR Data Processing Script
Bu skript PDF hesabatlarını emal edir və Dashboard üçün lazım olan bazaları yaradır.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm

from src.config import Config
from src.pdf_processor import DDRParser
from src.nlp_processor import NLPProcessor
from src.knowledge_graph import KnowledgeGraph
from src.rag_system import RAGSystem

# Logging nizamlanması
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def run_processing_pipeline():
    """Bütün emal prosesini ardıcıl yerinə yetirir"""
    
    # 1. Başlanğıc nizamlamaları
    pdf_files = list(Config.DATA_PATH.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"'{Config.DATA_PATH}' qovluğunda PDF faylı tapılmadı!")
        return

    logger.info(f"🚀 {len(pdf_files)} faylın emalına başlanılır...")
    
    # Komponentlərin inisializasiyası
    parser = DDRParser()
    nlp = NLPProcessor()
    kg = KnowledgeGraph()
    rag = RAGSystem()
    
    processed_results = []

    # 2. PDF Parsing və NLP Analizi
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Faylı strukturlaşdırılmış dataya çevir
            raw_data = parser.parse_pdf(pdf_path)
            
            # NLP ilə zənginləşdir (Xülasə, anomaliya, trendlər)
            enriched_data = nlp.process_ddr(raw_data)
            
            processed_results.append(enriched_data)
        except Exception as e:
            logger.warning(f"⚠️ {pdf_path.name} emal edilərkən xəta: {e}")

    # 3. Məlumatların Yadda Saxlanılması (JSON)
    output_path = Config.PROCESSED_DATA_PATH / "processed_ddrs.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Emal edilmiş data yadda saxlanıldı: {output_path}")

    # 4. Bilik Qrafının (Knowledge Graph) Qurulması
    logger.info("🕸️ Bilik Qrafı yaradılır...")
    for data in processed_results:
        kg.build_from_ddr(data)
    
    kg.visualize(output_file=str(Config.PROCESSED_DATA_PATH / "knowledge_graph.html"))
    logger.info("✅ Bilik Qrafı vizuallaşdırıldı.")

    # 5. RAG Vektor Bazasının Yenilənməsi
    logger.info("📚 RAG Vektor Bazası qurulur...")
    rag.add_documents(processed_results)
    logger.info("✅ RAG sistemi hazırdır.")

    # 6. Trend Analizi Faylı
    logger.info("📈 Trend Analizi aparılır...")
    trends = nlp.analyze_trends(processed_results)
    with open(Config.PROCESSED_DATA_PATH / "trends.json", 'w', encoding='utf-8') as f:
        json.dump(trends, f, indent=2, ensure_ascii=False)
    
    logger.info("\n✨ Bütün proses uğurla tamamlandı!")

if __name__ == "__main__":
    run_processing_pipeline()