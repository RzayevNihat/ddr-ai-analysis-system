import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
from src.config import Config
from src.llm_service import LLMService
import logging
import hashlib 
import pickle 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Retrieval Augmented Generation System for DDR data"""
    def __init__(self):
        self.cache_dir = Config.PROCESSED_DATA_PATH / "embedding_cache"
        self.cache_dir.mkdir(exist_ok=True)
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str):
        """Get embedding from cache if exists"""
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _cache_embedding(self, text: str, embedding):
        """Cache embedding"""
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search with caching"""
        # Check cache
        cached = self._get_cached_embedding(query)
        if cached is not None:
            query_embedding = cached
        else:
            query_embedding = self.embeddings_model.encode([query])[0].tolist()
            self._cache_embedding(query, query_embedding)
    def __init__(self):
        self.llm = LLMService()
        
        # Initialize embeddings model
        self.embeddings_model = SentenceTransformer(Config.EMBEDDINGS_MODEL)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(Config.CHROMA_PERSIST_DIR)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="ddr_reports",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("RAG System initialized")
    
    def add_documents(self, ddr_data_list: List[Dict[str, Any]]):
        """Add DDR documents to vector store"""
        documents = []
        metadatas = []
        ids = []
        
        for i, ddr in enumerate(ddr_data_list):
            # Create searchable text
            doc_text = self._create_document_text(ddr)
            
            # Metadata
            metadata = {
                'wellbore': ddr.get('wellbore', ''),
                'period': ddr.get('period', ''),
                'operator': ddr.get('operator', ''),
                'depth_md': str(ddr.get('depth_md', '')),
                'filename': ddr.get('filename', '')
            }
            
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(f"ddr_{i}_{ddr.get('filename', i)}")
        
        # Generate embeddings and add to collection
        embeddings = self.embeddings_model.encode(documents).tolist()
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def _create_document_text(self, ddr: Dict[str, Any]) -> str:
        """Create searchable text from DDR data"""
        text_parts = []
        
        # Header info
        text_parts.append(f"Wellbore: {ddr.get('wellbore', '')}")
        text_parts.append(f"Period: {ddr.get('period', '')}")
        text_parts.append(f"Operator: {ddr.get('operator', '')}")
        text_parts.append(f"Rig: {ddr.get('rig_name', '')}")
        text_parts.append(f"Current Depth MD: {ddr.get('depth_md', '')}m")
        text_parts.append(f"Hole Size: {ddr.get('hole_size', '')} inches")
        
        # Summary
        summary = ddr.get('summary', {})
        if summary.get('activities_24h'):
            text_parts.append(f"\nActivities: {summary['activities_24h']}")
        if summary.get('planned_24h'):
            text_parts.append(f"Planned: {summary['planned_24h']}")
        
        # Operations
        if ddr.get('operations'):
            text_parts.append("\nOperations:")
            for op in ddr['operations'][:10]:  # First 10
                text_parts.append(
                    f"- {op.get('start_time')}-{op.get('end_time')}: "
                    f"{op.get('activity')} at {op.get('depth')}m - {op.get('state')}"
                )
        
        # Lithology
        if ddr.get('lithology'):
            text_parts.append("\nLithology:")
            for lith in ddr['lithology']:
                text_parts.append(
                    f"- {lith.get('start_depth')}-{lith.get('end_depth')}m: "
                    f"{lith.get('description')}"
                )
        
        # Gas readings
        if ddr.get('gas_readings'):
            high_gas = [g for g in ddr['gas_readings'] if g.get('gas_percentage', 0) > 1.0]
            if high_gas:
                text_parts.append("\nGas Readings:")
                for gas in high_gas[:5]:
                    text_parts.append(
                        f"- {gas.get('depth')}m: {gas.get('gas_percentage')}% "
                        f"(C1: {gas.get('c1_ppm')} ppm)"
                    )
        
        return "\n".join(text_parts)
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        # Generate query embedding
        query_embedding = self.embeddings_model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def answer_question(self, question: str, n_context: int = 3) -> Dict[str, Any]:
        """Answer a question using RAG"""
        # Search for relevant context
        search_results = self.search(question, n_results=n_context)
        
        # Build context
        context_parts = []
        for i, result in enumerate(search_results):
            context_parts.append(f"--- Document {i+1} ---")
            context_parts.append(result['document'])
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Generate answer using LLM
        answer = self.llm.answer_question(question, context)
        
        return {
            'question': question,
            'answer': answer,
            'sources': [r['metadata'] for r in search_results],
            'context_used': len(search_results)
        }
    
    def query_by_filters(self, filters: Dict[str, str], n_results: int = 10) -> List[Dict]:
        """Query documents by metadata filters"""
        where_clause = {}
        
        if 'wellbore' in filters:
            where_clause['wellbore'] = filters['wellbore']
        if 'operator' in filters:
            where_clause['operator'] = filters['operator']
        
        results = self.collection.get(
            where=where_clause,
            limit=n_results
        )
        
        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                'id': results['ids'][i],
                'document': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return formatted_results
    
    def get_all_wellbores(self) -> List[str]:
        """Get list of all wellbores"""
        results = self.collection.get()
        wellbores = set()
        
        for metadata in results['metadatas']:
            if metadata.get('wellbore'):
                wellbores.add(metadata['wellbore'])
        
        return sorted(list(wellbores))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        count = self.collection.count()
        wellbores = self.get_all_wellbores()
        
        return {
            'total_documents': count,
            'unique_wellbores': len(wellbores),
            'wellbores': wellbores
        }