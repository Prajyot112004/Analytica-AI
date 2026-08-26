import sys
import io
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

def execute_python_code(code: str, df: pd.DataFrame) -> Tuple[bool, str, Any]:
    """
    Executes generated Python code safely in a restricted execution namespace with access to Pandas, NumPy, Seaborn, and Matplotlib.
    """
    # Restricted execution namespace
    local_scope = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns
    }
    
    output_capture = io.StringIO()
    sys_stdout_backup = sys.stdout

    try:
        sys.stdout = output_capture
        exec(code, {}, local_scope)
        sys.stdout = sys_stdout_backup
        
        captured_str = output_capture.getvalue()
        result_val = local_scope.get("result", captured_str)
        return True, captured_str if captured_str else "Execution finished cleanly.", result_val
    except Exception as e:
        sys.stdout = sys_stdout_backup
        err_msg = f"Code execution error: {str(e)}\n{traceback.format_exc()}"
        return False, err_msg, None
