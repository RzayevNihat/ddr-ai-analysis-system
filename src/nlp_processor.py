import re
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from collections import defaultdict
from src.config import Config
from src.llm_service import LLMService

class NLPProcessor:
    """Process DDR data using NLP techniques"""
    
    def __init__(self):
        self.llm = LLMService()
        self.activity_keywords = {
            'drilling': ['drill', 'drilled', 'drilling', 'hole'],
            'tripping': ['trip', 'tripping', 'pooh', 'rih', 'pull', 'run'],
            'circulating': ['circulate', 'circulation', 'circulating', 'pump'],
            'casing': ['casing', 'cement', 'cementing'],
            'testing': ['test', 'testing', 'function test', 'bop'],
            'reaming': ['ream', 'reamed', 'reaming', 'wash'],
            'stuck_pipe': ['stuck', 'stuck pipe', 'fish', 'fishing', 'jar'],
            'lost_circulation': ['lost circulation', 'losses', 'losing'],
            'survey': ['survey', 'surveying', 'mwd']
        }
    
    def classify_activity(self, text: str) -> str:
        """Classify drilling activity from text"""
        text_lower = text.lower()
        
        # Score each activity type
        scores = defaultdict(int)
        for activity, keywords in self.activity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[activity] += 1
        
        # Return activity with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'other'
    
    def detect_anomalies(self, ddr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in DDR data"""
        anomalies = []
        
        # Check for stuck pipe
        if ddr_data.get('operations'):
            for op in ddr_data['operations']:
                if 'stuck' in op.get('remark', '').lower():
                    anomalies.append({
                        'type': 'stuck_pipe',
                        'severity': 'high',
                        'time': op.get('start_time'),
                        'description': op.get('remark')
                    })
        
        # Check for lost circulation
        summary = ddr_data.get('summary', {})
        if 'lost' in str(summary).lower() and 'circulation' in str(summary).lower():
            anomalies.append({
                'type': 'lost_circulation',
                'severity': 'high',
                'description': summary.get('activities_24h', '')
            })
        
        # Check for high gas readings
        gas_readings = ddr_data.get('gas_readings', [])
        for reading in gas_readings:
            if reading.get('gas_percentage') and reading['gas_percentage'] > 1.2:
                anomalies.append({
                    'type': 'high_gas',
                    'severity': 'medium',
                    'depth': reading.get('depth'),
                    'value': reading.get('gas_percentage'),
                    'description': f"Gas peak of {reading['gas_percentage']}% at {reading['depth']}m"
                })
        
        # Check operations state
        if ddr_data.get('operations'):
            for op in ddr_data['operations']:
                if op.get('state') == 'fail':
                    anomalies.append({
                        'type': 'operation_failure',
                        'severity': 'high',
                        'time': op.get('start_time'),
                        'activity': op.get('activity'),
                        'description': op.get('remark')
                    })
        
        return anomalies
    
    def create_daily_summary(self, ddr_data: Dict[str, Any]) -> str:
        """Create a concise daily summary using LLM"""
        
        # Prepare context
        context = f"""
Wellbore: {ddr_data.get('wellbore', 'N/A')}
Period: {ddr_data.get('period', 'N/A')}
Operator: {ddr_data.get('operator', 'N/A')}
Rig: {ddr_data.get('rig_name', 'N/A')}

Current Depth (MD): {ddr_data.get('depth_md', 'N/A')}m
Hole Size: {ddr_data.get('hole_size', 'N/A')}"

Activities Summary:
{ddr_data.get('summary', {}).get('activities_24h', 'N/A')}

Operations:
"""
        
        # Add key operations
        if ddr_data.get('operations'):
            for op in ddr_data['operations'][:5]:  # First 5 operations
                context += f"- {op.get('start_time')}-{op.get('end_time')}: {op.get('activity')} at {op.get('depth')}m - {op.get('state')}\n"
        
        # Add anomalies
        anomalies = self.detect_anomalies(ddr_data)
        if anomalies:
            context += "\nAnomalies Detected:\n"
            for anomaly in anomalies[:3]:
                context += f"- {anomaly['type']}: {anomaly['description']}\n"
        
        # Generate summary using LLM
        prompt = f"""You are a drilling engineer. Create a concise 3-4 sentence summary of this Daily Drilling Report.
Focus on: key activities, depth progress, any issues/anomalies, and overall status.

{context}

Summary:"""
        
        summary = self.llm.generate_text(prompt, max_tokens=200)
        return summary
    
    def extract_parameters(self, ddr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key drilling parameters"""
        params = {
            'wellbore': ddr_data.get('wellbore'),
            'period': ddr_data.get('period'),
            'depth_md': ddr_data.get('depth_md'),
            'depth_tvd': ddr_data.get('depth_tvd'),
            'hole_size': ddr_data.get('hole_size'),
            'operator': ddr_data.get('operator'),
            'rig_name': ddr_data.get('rig_name'),
            'activities': [],
            'fluid_properties': [],
            'survey_points': [],
            'gas_readings': [],
            'lithology': [],
            'anomalies': []
        }
        
        # Extract activities
        if ddr_data.get('operations'):
            for op in ddr_data['operations']:
                params['activities'].append({
                    'time': f"{op.get('start_time')}-{op.get('end_time')}",
                    'type': self.classify_activity(op.get('remark', '')),
                    'depth': op.get('depth'),
                    'state': op.get('state')
                })
        
        # Extract fluid properties
        if ddr_data.get('drilling_fluid'):
            for fluid in ddr_data['drilling_fluid']:
                if fluid.get('density'):
                    params['fluid_properties'].append(fluid)
        
        # Extract survey data
        if ddr_data.get('survey_data'):
            params['survey_points'] = ddr_data['survey_data']
        
        # Extract gas readings
        if ddr_data.get('gas_readings'):
            params['gas_readings'] = ddr_data['gas_readings']
        
        # Extract lithology
        if ddr_data.get('lithology'):
            params['lithology'] = ddr_data['lithology']
        
        # Extract anomalies
        params['anomalies'] = self.detect_anomalies(ddr_data)
        
        return params
    
    def analyze_trends(self, all_ddr_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends across multiple DDRs"""
        
        # Create time series data
        depth_progress = []
        gas_trends = []
        rop_data = []
        anomaly_timeline = []
        
        for ddr in all_ddr_data:
            period = ddr.get('period', '')
            depth_md = ddr.get('depth_md')
            
            if depth_md:
                depth_progress.append({
                    'date': self._extract_date(period),
                    'depth_md': depth_md,
                    'wellbore': ddr.get('wellbore')
                })
            
            # Gas readings trend
            for gas in ddr.get('gas_readings', []):
                gas_trends.append({
                    'date': self._extract_date(period),
                    'depth': gas.get('depth'),
                    'percentage': gas.get('gas_percentage'),
                    'wellbore': ddr.get('wellbore')
                })
            
            # Anomalies timeline
            anomalies = self.detect_anomalies(ddr)
            for anomaly in anomalies:
                anomaly_timeline.append({
                    'date': self._extract_date(period),
                    'type': anomaly['type'],
                    'severity': anomaly['severity'],
                    'wellbore': ddr.get('wellbore')
                })
        
        return {
            'depth_progress': depth_progress,
            'gas_trends': gas_trends,
            'anomaly_timeline': anomaly_timeline
        }
    
    def _extract_date(self, period_str: str) -> str:
        """Extract date from period string"""
        # Period format: "1997-11-13 00:00 - 1997-11-14 00:00"
        match = re.search(r'(\d{4}-\d{2}-\d{2})', period_str)
        return match.group(1) if match else ""
    
    def classify_events(self, ddr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Classify all events in a DDR"""
        events = []
        
        if ddr_data.get('operations'):
            for op in ddr_data['operations']:
                event = {
                    'start_time': op.get('start_time'),
                    'end_time': op.get('end_time'),
                    'depth': op.get('depth'),
                    'activity_type': self.classify_activity(op.get('remark', '')),
                    'state': op.get('state'),
                    'description': op.get('remark'),
                    'duration': self._calculate_duration(op.get('start_time'), op.get('end_time'))
                }
                events.append(event)
        
        return events
    
    def _calculate_duration(self, start: str, end: str) -> float:
        """Calculate duration in hours"""
        try:
            if start and end and ':' in start and ':' in end:
                start_h, start_m = map(int, start.split(':'))
                end_h, end_m = map(int, end.split(':'))
                
                start_minutes = start_h * 60 + start_m
                end_minutes = end_h * 60 + end_m
                
                if end_minutes < start_minutes:
                    end_minutes += 24 * 60
                
                return (end_minutes - start_minutes) / 60.0
        except:
            pass
        return 0.0