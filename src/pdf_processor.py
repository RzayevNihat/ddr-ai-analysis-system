import re
import pdfplumber
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DDRParser:
    """Parse Daily Drilling Reports from PDFs"""
    
    def __init__(self):
        self.section_patterns = {
            'summary': r'Summar\s*y\s*repor\s*t',
            'operations': r'Operations',
            'drilling_fluid': r'Drilling\s*Fluid',
            'survey': r'Sur\s*vey\s*Station',
            'lithology': r'Lithology\s*Infor\s*mation',
            'gas': r'Gas\s*Reading\s*Infor\s*mation',
            'pore_pressure': r'Pore\s*Pressure',
            'stratigraphic': r'Stratigraphic\s*Infor\s*mation'
        }
    
    def parse_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Parse a single DDR PDF file"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n"
            
            # Extract with ROBUST methods
            wellbore = self._extract_wellbore_robust(full_text, pdf_path.name)
            period = self._extract_period_robust(full_text)
            operator = self._extract_operator_robust(full_text)
            
            data = {
                'filename': pdf_path.name,
                'wellbore': wellbore,
                'period': period,
                'operator': operator,
                'rig_name': self._extract_field(full_text, r'Rig\s*Name:\s*([^\n]+)'),
                'summary': self._extract_summary(full_text),
                'operations': self._extract_operations(full_text),
                'drilling_fluid': self._extract_drilling_fluid(full_text),
                'survey_data': self._extract_survey(full_text),
                'lithology': self._extract_lithology(full_text),
                'gas_readings': self._extract_gas_readings(full_text),
                'depth_md': self._extract_depth(full_text, 'Depth mMd'),
                'depth_tvd': self._extract_depth(full_text, 'Depth mTVD'),
                'hole_size': self._extract_field(full_text, r'Hole\s*Dia\s*\(in\):\s*([0-9.]+)'),
                'raw_text': full_text
            }
            
            logger.info(f"✓ {pdf_path.name}: {wellbore} | {period[:20]}... | {operator}")
            return data
            
        except Exception as e:
            logger.error(f"✗ Error parsing {pdf_path}: {str(e)}")
            return {'filename': pdf_path.name, 'error': str(e)}
    
    def _extract_wellbore_robust(self, text: str, filename: str) -> str:
        """Extract wellbore - handles duplicate 'Wellbore:Wellbore:' format"""
        # Pattern 1: Direct "Wellbore:Wellbore: 15/9-19 A" (common in these PDFs)
        patterns = [
            r'Wellbore:\s*Wellbore:\s*([^\s\n]+(?:\s+[A-Z0-9])?)',  # Duplicate format
            r'Wellbore:\s*([^\s\n]+(?:\s+[A-Z0-9])?)',
            r'Well\s*bore:\s*([^\s\n]+(?:\s+[A-Z0-9])?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                wellbore = match.group(1).strip()
                if wellbore and len(wellbore) > 2:
                    return wellbore
        
        # Pattern 2: Direct format like "15/9-19 B"
        match = re.search(r'\b(\d{1,2}/\d{1,2}-[A-Z0-9-]+(?:\s+[A-Z0-9])?)\b', text)
        if match:
            return match.group(1).strip()
        
        # Pattern 3: From filename
        filename_clean = filename.replace('.pdf', '')
        
        # "15_9_19_A_date.pdf" format
        match = re.search(r'(\d{1,2})_(\d{1,2})_(\d+)_([A-Z0-9]+)', filename_clean)
        if match:
            block, field, well, suffix = match.groups()
            if not suffix.isdigit():  # Not a date
                return f"{block}/{field}-{well} {suffix}"
            return f"{block}/{field}-{well}"
        
        # "15_9_F_10_date.pdf" format
        match = re.search(r'(\d{1,2})_(\d{1,2})_([A-Z])_(\d+)', filename_clean)
        if match:
            return f"{match.group(1)}/{match.group(2)}-{match.group(3)}-{match.group(4)}"
        
        logger.warning(f"Could not extract wellbore from {filename}")
        return filename_clean[:30]
    
    def _extract_period_robust(self, text: str) -> str:
        """Extract period - handles duplicate 'Period:Period:' format"""
        patterns = [
            # Pattern 1: "Period:Period: 1997-12-04 00:00 - 1997-12-05 00:00"
            r'Period:\s*Period:\s*([^\n]+)',
            # Pattern 2: Normal "Period: ..."
            r'Period:\s*([^\n]+)',
            # Pattern 3: Direct date range
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*-\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                period = match.group(1).strip()
                # Clean up any trailing junk after the date range
                period = re.sub(r'(Status|Report|Normal).*$', '', period, flags=re.IGNORECASE).strip()
                if period and len(period) > 10:
                    return period
        
        return ""
    
    def _extract_operator_robust(self, text: str) -> str:
        """Extract operator - clean up trailing content"""
        patterns = [
            r'Operator:\s*([A-Za-z\s]+)(?:\n|Rig|Formation|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                operator = match.group(1).strip()
                # Remove common trailing words
                operator = re.sub(r'\s*(Formation|Rig|Drilling|Contractor).*$', '', operator, flags=re.IGNORECASE)
                operator = operator.strip()
                if operator and len(operator) > 2:
                    return operator
        
        return ""
    
    def _extract_field(self, text: str, pattern: str) -> str:
        """Extract a field using regex pattern"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            # Stop at first newline
            result = result.split('\n')[0].strip()
            return result
        return ""
    
    def _extract_summary(self, text: str) -> Dict[str, str]:
        """Extract summary of activities"""
        summary = {}
        
        # Extract summary of activities (24 hours)
        match = re.search(r'Summar\s*y\s*of\s*activities\s*\(24\s*Hours\)\s*\n([^\n]+(?:\n(?!Summar)[^\n]+)*)', 
                         text, re.IGNORECASE)
        if match:
            summary['activities_24h'] = match.group(1).strip()
        
        # Extract planned activities
        match = re.search(r'Summar\s*y\s*of\s*planned\s*activities\s*\(24\s*Hours\)\s*\n([^\n]+(?:\n(?!Operations)[^\n]+)*)', 
                         text, re.IGNORECASE)
        if match:
            summary['planned_24h'] = match.group(1).strip()
        
        return summary
    
    def _extract_operations(self, text: str) -> List[Dict[str, str]]:
        """Extract operations data"""
        operations = []
        
        # Find operations section
        ops_match = re.search(r'Operations\s*\n.*?StartStart.*?\n(.*?)(?=\n\n[A-Z]|\nDrilling\s*Fluid|\Z)', 
                            text, re.DOTALL | re.IGNORECASE)
        
        if ops_match:
            ops_text = ops_match.group(1)
            lines = [l.strip() for l in ops_text.split('\n') if l.strip()]
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    op = {
                        'start_time': parts[0] if ':' in parts[0] else '',
                        'end_time': parts[1] if ':' in parts[1] else '',
                        'depth': self._extract_number(line),
                        'activity': self._extract_activity(line),
                        'state': self._extract_state(line),
                        'remark': line
                    }
                    operations.append(op)
        
        return operations
    
    def _extract_activity(self, text: str) -> str:
        """Extract activity type from text"""
        text_lower = text.lower()
        activities = {
            'drilling': ['drilling', 'drill', 'drilled'],
            'trip': ['trip', 'tripping', 'pooh', 'rih'],
            'circulating': ['circulating', 'circulation'],
            'casing': ['casing', 'run casing'],
            'bop': ['bop', 'blowout'],
            'fishing': ['fishing', 'fish'],
            'reaming': ['reaming', 'ream'],
            'testing': ['testing', 'test'],
            'completion': ['completion', 'completing']
        }
        
        for activity, keywords in activities.items():
            if any(kw in text_lower for kw in keywords):
                return activity
        return "other"
    
    def _extract_state(self, text: str) -> str:
        """Extract state (ok, fail) from text"""
        if 'fail' in text.lower():
            return 'fail'
        elif 'ok' in text.lower():
            return 'ok'
        return 'unknown'
    
    def _extract_drilling_fluid(self, text: str) -> List[Dict[str, Any]]:
        """Extract drilling fluid data"""
        fluids = []
        
        fluid_match = re.search(r'Drilling\s*Fluid\s*\n(.*?)(?=\n\n[A-Z]|\nPore\s*Pressure|\Z)', 
                              text, re.DOTALL | re.IGNORECASE)
        
        if fluid_match:
            fluid_text = fluid_match.group(1)
            
            density_matches = re.findall(r'Fluid\s*Density\s*\(g/cm3\)\s*\n([0-9.]+)', fluid_text)
            visc_matches = re.findall(r'Funnel\s*Visc\s*\(s\)\s*\n([0-9.]+)', fluid_text)
            type_matches = re.findall(r'Fluid\s*Type\s*\n([^\n]+)', fluid_text)
            
            for i in range(max(len(density_matches), len(visc_matches), len(type_matches))):
                fluid = {
                    'density': float(density_matches[i]) if i < len(density_matches) else None,
                    'viscosity': float(visc_matches[i]) if i < len(visc_matches) else None,
                    'type': type_matches[i].strip() if i < len(type_matches) else None
                }
                fluids.append(fluid)
        
        return fluids
    
    def _extract_survey(self, text: str) -> List[Dict[str, float]]:
        """Extract survey station data"""
        surveys = []
        
        survey_match = re.search(r'Sur\s*vey\s*Station\s*\n.*?Depth\s*mMD.*?\n(.*?)(?=\n\n[A-Z]|\Z)', 
                               text, re.DOTALL | re.IGNORECASE)
        
        if survey_match:
            survey_text = survey_match.group(1)
            lines = [l.strip() for l in survey_text.split('\n') if l.strip()]
            
            for line in lines:
                numbers = re.findall(r'[0-9]+\.?[0-9]*', line)
                if len(numbers) >= 4:
                    surveys.append({
                        'depth_md': float(numbers[0]),
                        'depth_tvd': float(numbers[1]),
                        'inclination': float(numbers[2]),
                        'azimuth': float(numbers[3])
                    })
        
        return surveys
    
    def _extract_lithology(self, text: str) -> List[Dict[str, Any]]:
        """Extract lithology information"""
        lithology = []
        
        lith_match = re.search(r'Lithology\s*Infor\s*mation\s*\n(.*?)(?=\n\n[A-Z]|\nGas\s*Reading|\Z)', 
                             text, re.DOTALL | re.IGNORECASE)
        
        if lith_match:
            lith_text = lith_match.group(1)
            lines = [l.strip() for l in lith_text.split('\n') if l.strip() and 'Start Depth' not in l]
            
            for line in lines:
                numbers = re.findall(r'[0-9]+\.?[0-9]*', line)
                if len(numbers) >= 2:
                    lithology.append({
                        'start_depth': float(numbers[0]),
                        'end_depth': float(numbers[1]),
                        'description': re.sub(r'[0-9]+\.?[0-9]*', '', line).strip()
                    })
        
        return lithology
    
    def _extract_gas_readings(self, text: str) -> List[Dict[str, Any]]:
        """Extract gas reading information"""
        gas_readings = []
        
        gas_match = re.search(r'Gas\s*Reading\s*Infor\s*mation\s*\n(.*?)(?=\Z)', 
                            text, re.DOTALL | re.IGNORECASE)
        
        if gas_match:
            gas_text = gas_match.group(1)
            lines = [l.strip() for l in gas_text.split('\n') if l.strip() and 'Time' not in l and 'Class' not in l]
            
            for line in lines:
                numbers = re.findall(r'[0-9]+\.?[0-9]*', line)
                if len(numbers) >= 3:
                    gas_readings.append({
                        'depth': float(numbers[0]) if numbers else None,
                        'gas_percentage': float(numbers[-5]) if len(numbers) >= 5 else None,
                        'c1_ppm': float(numbers[-4]) if len(numbers) >= 4 else None,
                        'c2_ppm': float(numbers[-3]) if len(numbers) >= 3 else None,
                        'class': 'peak' if 'peak' in line.lower() else 'trip'
                    })
        
        return gas_readings
    
    def _extract_depth(self, text: str, depth_type: str) -> float:
        """Extract depth value"""
        pattern = f'{depth_type}:\\s*([0-9]+\\.?[0-9]*)'
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        return None
    
    def _extract_number(self, text: str) -> float:
        """Extract first number from text"""
        match = re.search(r'[0-9]+\.?[0-9]*', text)
        return float(match.group(0)) if match else None
    
    def parse_all_pdfs(self, pdf_dir: Path) -> List[Dict[str, Any]]:
        """Parse all PDFs in a directory"""
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        all_data = []
        for pdf_path in pdf_files:
            data = self.parse_pdf(pdf_path)
            all_data.append(data)
        
        return all_data