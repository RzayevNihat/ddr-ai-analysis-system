import os
from groq import Groq
from src.config import Config
import logging
import time
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Groq API rate limit - BALANCED"""
    
    def __init__(self, rpm_limit=28, tpm_limit=16000):
        # ✅ BALANCED: Nə çox konservativ, nə də aqressiv
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_times = []
        self.token_usage = []
        self.total_requests = 0
        self.total_tokens = 0
        self.total_wait_time = 0
        self.rate_limit_hits = 0
        self.last_request_time = None
    
    def _clean_old_entries(self):
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > cutoff]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff]
    
    def _get_current_rpm(self):
        self._clean_old_entries()
        return len(self.request_times)
    
    def _get_current_tpm(self):
        self._clean_old_entries()
        return sum(tokens for _, tokens in self.token_usage)
    
    def _estimate_tokens(self, prompt: str, max_tokens: int) -> int:
        input_tokens = len(prompt) // 4
        return input_tokens + max_tokens
    
    def wait_if_needed(self, prompt: str, max_tokens: int):
        """PROACTIVE rate limiting - balanced"""
        estimated_tokens = self._estimate_tokens(prompt, max_tokens)
        
        # ✅ Minimum 2.5s delay (3s yox, amma kifayət)
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            min_delay = 2.5
            if elapsed < min_delay:
                wait = min_delay - elapsed
                time.sleep(wait)
        
        while True:
            self._clean_old_entries()
            current_rpm = self._get_current_rpm()
            current_tpm = self._get_current_tpm()
            
            # ✅ BALANCED buffer (3 RPM, 1500 TPM)
            rpm_available = current_rpm < self.rpm_limit - 3
            tpm_available = (current_tpm + estimated_tokens) < self.tpm_limit - 1500
            
            if rpm_available and tpm_available:
                self.request_times.append(datetime.now())
                self.last_request_time = datetime.now()
                self.total_requests += 1
                break
            else:
                wait_time = 4  # 5s yox, 4s
                
                if not rpm_available:
                    logger.warning(f"⏸️ RPM limit yaxın ({current_rpm}/{self.rpm_limit}). {wait_time}s gözləyir")
                    self.rate_limit_hits += 1
                
                if not tpm_available:
                    logger.warning(f"⏸️ TPM limit yaxın ({current_tpm}/{self.tpm_limit}). {wait_time}s gözləyir")
                    self.rate_limit_hits += 1
                
                time.sleep(wait_time)
                self.total_wait_time += wait_time
    
    def record_tokens(self, token_count: int):
        self.token_usage.append((datetime.now(), token_count))
        self.total_tokens += token_count
    
    def get_stats(self):
        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'total_wait_time': self.total_wait_time,
            'rate_limit_hits': self.rate_limit_hits,
            'current_rpm': self._get_current_rpm(),
            'current_tpm': self._get_current_tpm()
        }


class LLMService:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.GROQ_MODEL
        self.rate_limiter = RateLimiter(rpm_limit=28, tpm_limit=16000)
        self.system_prompt = """Sən gündəlik qazma hesabatları (DDR), quyu əməliyyatları və neft mühəndisliyi üzrə dərin biliyə malik ekspert qazma mühəndisisən.

VACİB: Giriş dilindən asılı olmayaraq HƏMİŞƏ Azərbaycan dilində cavab ver."""
    
    def generate_text(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        max_retries = 5
        base_delay = 8  # 10 yox, 8
        
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed(prompt, max_tokens)
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if hasattr(chat_completion, 'usage'):
                    self.rate_limiter.record_tokens(chat_completion.usage.total_tokens)
                
                return chat_completion.choices[0].message.content
            
            except Exception as e:
                error_str = str(e).lower()
                
                if "429" in error_str or "rate_limit" in error_str:
                    # ✅ Balanced exponential backoff
                    wait_time = base_delay * (1.8 ** attempt)
                    logger.warning(f"⚠️ Rate limit hit! {wait_time:.0f}s gözləyir... (Cəhd {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                if attempt == max_retries - 1:
                    logger.error(f"❌ API xətası: {str(e)}")
                    return f"Xəta: {str(e)}"
                else:
                    logger.warning(f"Retry {attempt + 1}")
                    time.sleep(base_delay)
        
        return "Xəta: Cavab alına bilmədi"
    
    def get_rate_limit_stats(self):
        return self.rate_limiter.get_stats()
    
    def answer_question(self, question: str, context: str) -> str:
        prompt = f"""Aşağıdakı qazma hesabatı məlumatlarına əsaslanaraq sualı Azərbaycan dilində cavablandır.

Kontekst: {context}
Sual: {question}

Cavab (Azərbaycanca):"""
        return self.generate_text(prompt, max_tokens=1024)
    
    def summarize_operations(self, operations_text: str) -> str:
        prompt = f"""Aşağıdakı qazma əməliyyatlarını Azərbaycan dilində 2-3 cümlə ilə xülasə et.

Əməliyyatlar: {operations_text}

Xülasə (Azərbaycanca):"""
        return self.generate_text(prompt, max_tokens=1024)
    
    def create_daily_summary(self, ddr_data: dict) -> str:
        activities = ddr_data.get('summary', {}).get('activities_24h', '')
        depth = ddr_data.get('depth_md', 'N/A')
        operations = ddr_data.get('operations', [])
        ops_text = "\n".join([op.get('remark', '') for op in operations[:10]])
        
        prompt = f"""Bu gündəlik qazma hesabatının Azərbaycan dilində 3-4 cümləlik xülasəsini hazırla.

Dərinlik (MD): {depth}m
Fəaliyyətlər: {activities}
Əməliyyatlar: {ops_text}

Gündəlik Xülasə (Azərbaycanca):"""
        
        return self.generate_text(prompt, max_tokens=1024, temperature=0.2)

