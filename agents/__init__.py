# -*- coding: utf-8 -*-
"""
Agents module - multi-agent system
For backward compatibility, you can still import from agents.py directly
"""

# This module imports from the original agents.py file
# which is in the same directory (now renamed but kept for compatibility)
__all__ = []

# Add the parent directory to sys.path to import the original agents module
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add V1 directory to path to import agents.py
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Import from original agents.py (which is now in V1 directory)
    # We use importlib to avoid confusion with the directory named 'agents'
    import importlib.util
    
    # First try to import from current directory (agents.py)
    agents_py_path = os.path.join(parent_dir, 'V1', 'agents.py')
    if os.path.exists(agents_py_path):
        spec = importlib.util.spec_from_file_location('agents_module', agents_py_path)
        agents_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agents_module)
        
        # Import symbols from the loaded module
        Agent = getattr(agents_module, 'Agent', None)
        ConceptExplanationAgent = getattr(agents_module, 'ConceptExplanationAgent', None)
        RequirementsAnalysisAgent = getattr(agents_module, 'RequirementsAnalysisAgent', None)
        SoftwareDesignAgent = getattr(agents_module, 'SoftwareDesignAgent', None)
        SoftwareTestingAgent = getattr(agents_module, 'SoftwareTestingAgent', None)
        ProjectManagementAgent = getattr(agents_module, 'ProjectManagementAgent', None)
        CodeImplementationAgent = getattr(agents_module, 'CodeImplementationAgent', None)
        SoftwareEthicsAgent = getattr(agents_module, 'SoftwareEthicsAgent', None)
        AgentCoordinator = getattr(agents_module, 'AgentCoordinator', None)
        select_agents_function = getattr(agents_module, 'select_agents_function', None)
        synthesize_answers_function = getattr(agents_module, 'synthesize_answers_function', None)
        
        __all__ = []
        for name, val in locals().items():
            if not name.startswith('_') and val is not None and name not in ['sys', 'os', 'importlib']:
                __all__.append(name)
        
        # Remove internal names from __all__
        __all__ = [name for name in __all__ if name not in ['current_dir', 'parent_dir', 'spec', 'agents_module', 'agents_py_path']]
except Exception as e:
    # If everything fails, we'll define a minimal fallback
    import warnings
    warnings.warn(f"Failed to import agents module: {e}")
    __all__ = []
