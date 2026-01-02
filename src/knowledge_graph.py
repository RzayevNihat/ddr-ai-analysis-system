import networkx as nx
from typing import List, Dict, Any
import json
from pyvis.network import Network
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Knowledge Graph for DDR data"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_types = {
            'wellbore': [],
            'activity': [],
            'depth': [],
            'formation': [],
            'lithology': [],
            'fluid': [],
            'anomaly': []
        }
    
    def add_wellbore_node(self, wellbore_name: str, properties: Dict = None):
        """Add wellbore node"""
        self.graph.add_node(
            wellbore_name,
            node_type='wellbore',
            **properties if properties else {}
        )
        self.node_types['wellbore'].append(wellbore_name)
    
    def add_activity_node(self, activity_id: str, activity_type: str, properties: Dict = None):
        """Add activity node"""
        self.graph.add_node(
            activity_id,
            node_type='activity',
            activity_type=activity_type,
            **properties if properties else {}
        )
        self.node_types['activity'].append(activity_id)
    
    def add_depth_node(self, depth: float, wellbore: str):
        """Add depth node"""
        depth_id = f"{wellbore}_depth_{depth}"
        self.graph.add_node(
            depth_id,
            node_type='depth',
            depth=depth,
            wellbore=wellbore
        )
        self.node_types['depth'].append(depth_id)
        return depth_id
    
    def add_formation_node(self, formation_name: str, properties: Dict = None):
        """Add formation node"""
        self.graph.add_node(
            formation_name,
            node_type='formation',
            **properties if properties else {}
        )
        self.node_types['formation'].append(formation_name)
    
    def add_lithology_node(self, lithology_id: str, description: str, properties: Dict = None):
        """Add lithology node"""
        self.graph.add_node(
            lithology_id,
            node_type='lithology',
            description=description,
            **properties if properties else {}
        )
        self.node_types['lithology'].append(lithology_id)
    
    def add_fluid_node(self, fluid_id: str, properties: Dict):
        """Add fluid node"""
        self.graph.add_node(
            fluid_id,
            node_type='fluid',
            **properties
        )
        self.node_types['fluid'].append(fluid_id)
    
    def add_anomaly_node(self, anomaly_id: str, anomaly_type: str, properties: Dict = None):
        """Add anomaly node"""
        self.graph.add_node(
            anomaly_id,
            node_type='anomaly',
            anomaly_type=anomaly_type,
            **properties if properties else {}
        )
        self.node_types['anomaly'].append(anomaly_id)
    
    def add_temporal_edge(self, from_node: str, to_node: str, relationship: str = 'NEXT'):
        """Add temporal relationship"""
        self.graph.add_edge(from_node, to_node, relationship=relationship, edge_type='temporal')
    
    def add_spatial_edge(self, from_node: str, to_node: str, relationship: str = 'AT_DEPTH'):
        """Add spatial relationship"""
        self.graph.add_edge(from_node, to_node, relationship=relationship, edge_type='spatial')
    
    def add_causal_edge(self, cause_node: str, effect_node: str, relationship: str = 'CAUSED'):
        """Add causal relationship"""
        self.graph.add_edge(cause_node, effect_node, relationship=relationship, edge_type='causal')
    
    def build_from_ddr(self, ddr_data: Dict[str, Any]):
        """Build knowledge graph from DDR data"""
        wellbore = ddr_data.get('wellbore', 'unknown')
        period = ddr_data.get('period', '')
        
        # Add wellbore node
        self.add_wellbore_node(
            wellbore,
            {
                'operator': ddr_data.get('operator'),
                'rig': ddr_data.get('rig_name'),
                'depth_md': ddr_data.get('depth_md'),
                'depth_tvd': ddr_data.get('depth_tvd')
            }
        )
        
        # Add activities
        prev_activity = None
        if ddr_data.get('operations'):
            for i, op in enumerate(ddr_data['operations']):
                activity_id = f"{wellbore}_activity_{i}_{op.get('start_time', '')}"
                
                self.add_activity_node(
                    activity_id,
                    op.get('activity', 'other'),
                    {
                        'start_time': op.get('start_time'),
                        'end_time': op.get('end_time'),
                        'depth': op.get('depth'),
                        'state': op.get('state'),
                        'remark': op.get('remark')
                    }
                )
                
                # Connect to wellbore
                self.graph.add_edge(wellbore, activity_id, relationship='HAS_ACTIVITY')
                
                # Temporal link
                if prev_activity:
                    self.add_temporal_edge(prev_activity, activity_id)
                
                # Connect to depth
                if op.get('depth'):
                    depth_id = self.add_depth_node(op['depth'], wellbore)
                    self.add_spatial_edge(activity_id, depth_id)
                
                prev_activity = activity_id
        
        # Add lithology
        if ddr_data.get('lithology'):
            for i, lith in enumerate(ddr_data['lithology']):
                lith_id = f"{wellbore}_lith_{i}"
                self.add_lithology_node(
                    lith_id,
                    lith.get('description', ''),
                    {
                        'start_depth': lith.get('start_depth'),
                        'end_depth': lith.get('end_depth')
                    }
                )
                
                # Connect to wellbore
                self.graph.add_edge(wellbore, lith_id, relationship='HAS_LITHOLOGY')
                
                # Connect to depth range
                if lith.get('start_depth'):
                    depth_id = self.add_depth_node(lith['start_depth'], wellbore)
                    self.graph.add_edge(lith_id, depth_id, relationship='FROM_DEPTH')
        
        # Add gas readings
        if ddr_data.get('gas_readings'):
            for i, gas in enumerate(ddr_data['gas_readings']):
                if gas.get('gas_percentage', 0) > 1.2:  # High gas
                    anomaly_id = f"{wellbore}_gas_anomaly_{i}"
                    self.add_anomaly_node(
                        anomaly_id,
                        'high_gas',
                        {
                            'gas_percentage': gas.get('gas_percentage'),
                            'depth': gas.get('depth'),
                            'c1_ppm': gas.get('c1_ppm')
                        }
                    )
                    
                    self.graph.add_edge(wellbore, anomaly_id, relationship='HAS_ANOMALY')
                    
                    if gas.get('depth'):
                        depth_id = self.add_depth_node(gas['depth'], wellbore)
                        self.add_spatial_edge(anomaly_id, depth_id)
        
        # Add drilling fluid
        if ddr_data.get('drilling_fluid'):
            for i, fluid in enumerate(ddr_data['drilling_fluid']):
                if fluid.get('density'):
                    fluid_id = f"{wellbore}_fluid_{i}"
                    self.add_fluid_node(fluid_id, fluid)
                    self.graph.add_edge(wellbore, fluid_id, relationship='USES_FLUID')
    
    def query_gas_peaks(self, threshold: float = 1.2) -> List[Dict[str, Any]]:
        """Query all intervals with gas peaks above threshold"""
        results = []
        
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'anomaly' and data.get('anomaly_type') == 'high_gas':
                if data.get('gas_percentage', 0) > threshold:
                    results.append({
                        'id': node,
                        'depth': data.get('depth'),
                        'gas_percentage': data.get('gas_percentage'),
                        'wellbore': self._get_wellbore(node)
                    })
        
        return sorted(results, key=lambda x: x.get('depth', 0))
    
    def query_lithology_at_depth(self, wellbore: str, depth: float) -> List[str]:
        """Query lithology at specific depth"""
        results = []
        
        for node, data in self.graph.nodes(data=True):
            if (data.get('node_type') == 'lithology' and 
                data.get('start_depth', 0) <= depth <= data.get('end_depth', float('inf'))):
                if wellbore in node:
                    results.append(data.get('description', ''))
        
        return results
    
    def query_activities_at_depth(self, wellbore: str, depth: float, tolerance: float = 10) -> List[Dict]:
        """Query activities at specific depth"""
        results = []
        
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'activity' and wellbore in node:
                activity_depth = data.get('depth')
                if activity_depth and abs(activity_depth - depth) <= tolerance:
                    results.append({
                        'activity': data.get('activity_type'),
                        'depth': activity_depth,
                        'time': f"{data.get('start_time')}-{data.get('end_time')}",
                        'state': data.get('state')
                    })
        
        return results
    
    def _get_wellbore(self, node_id: str) -> str:
        """Extract wellbore from node ID"""
        # Find connected wellbore
        for neighbor in self.graph.predecessors(node_id):
            if self.graph.nodes[neighbor].get('node_type') == 'wellbore':
                return neighbor
        return 'unknown'
    
    def visualize(self, output_file: str = "knowledge_graph.html"):
        """Visualize knowledge graph"""
        net = Network(height="750px", width="100%", directed=True)
        
        # Color mapping for node types
        colors = {
            'wellbore': '#FF6B6B',
            'activity': '#4ECDC4',
            'depth': '#95E1D3',
            'formation': '#F38181',
            'lithology': '#AA96DA',
            'fluid': '#FCBAD3',
            'anomaly': '#FF0000'
        }
        
        # Add nodes with colors
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'other')
            color = colors.get(node_type, '#CCCCCC')
            title = f"{node_type}: {node}\n" + "\n".join([f"{k}: {v}" for k, v in data.items()])
            
            net.add_node(node, label=node, color=color, title=title)
        
        # Add edges
        for source, target, data in self.graph.edges(data=True):
            net.add_edge(source, target, label=data.get('relationship', ''))
        
        net.write_html(output_file)
        logger.info(f"Knowledge graph saved to {output_file}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get graph statistics"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'wellbores': len(self.node_types['wellbore']),
            'activities': len(self.node_types['activity']),
            'depths': len(self.node_types['depth']),
            'lithologies': len(self.node_types['lithology']),
            'anomalies': len(self.node_types['anomaly'])
        }
    # knowledge_graph.py-ə əlavə:
    def query_core_samples(self) -> List[Dict]:
        """Query: When were core samples taken and what were lithologies"""
        results = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'activity':
                if 'core' in str(data.get('remark', '')).lower():
                    # Get connected lithology
                    lithologies = self._get_connected_lithology(node)
                    results.append({
                        'time': data.get('start_time'),
                        'depth': data.get('depth'),
                        'lithology': lithologies
                    })
        return results
    
    def query_core_samples(self) -> List[Dict]:
        """Query: When were core samples taken and what were lithologies"""
        results = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'activity':
                # 'core' sözü keçən fəaliyyətləri tap
                remark = str(data.get('remark', '')).lower()
                activity = str(data.get('activity_type', '')).lower()
            
                if 'core' in remark or 'core' in activity:
                    wellbore = self._get_wellbore(node)
                    depth = data.get('depth')
                
                    # Lithology tap
                    lithologies = []
                    if wellbore and depth:
                        lithologies = self.query_lithology_at_depth(wellbore, depth)
                
                    results.append({
                        'time': data.get('start_time'),
                        'depth': depth,
                        'lithology': lithologies,
                        'wellbore': wellbore
                    })
        return results
