import numpy as np
import pandas as pd

import streamlit as st

from gym_pandemic.envs.pandemic_env import PandemicEnv
# from gym_pandemic.envs.pandemic_immunity_env import PandemicImmunityEnv


st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))
