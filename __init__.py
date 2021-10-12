import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from bks_pypack.df_preprocess import df_basic as dbsc
from bks_pypack.df_preprocess import preprocess_for_plotly as pfp