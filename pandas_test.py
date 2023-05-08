import streamlit as st
import numpy as np
import pandas as pd

df = pd.DataFrame(
    np.random.rand(10, 3),
    columns=['a', 'b', 'c']
    
)
print(df)