import streamlit as st
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor
from io import StringIO

# Function to execute the notebook and capture the output
def execute_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_content = nbformat.read(f, as_version=4)

    # Prepare the notebook for execution
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    # Execute the notebook
    ep.preprocess(nb_content, {'metadata': {'path': './'}})

    # Capture the output of the notebook execution
    output = ''
    for cell in nb_content.cells:
        if 'outputs' in cell:
            for output_item in cell['outputs']:
                if 'text/plain' in output_item.data:
                    output += output_item.data['text/plain'] + '\n'
    return output

# Streamlit app layout
st.title("Run Jupyter Notebook and Display Output")
st.write(
    """
    This Streamlit app executes a Jupyter notebook and shows its output below.
    """
)

# Select a notebook
# notebook_file = st.text_input('Enter the path of the notebook to execute:', 'your_notebook.ipynb')
notebook_file = 'movielens_explicit.ipynb'

# If a valid notebook path is provided, execute and display the output
if notebook_file:
    try:
        # Execute the notebook and get the output
        output = execute_notebook(notebook_file)
        st.text_area("Notebook Output:", value=output, height=300)
    except Exception as e:
        st.error(f"Error executing the notebook: {str(e)}")
