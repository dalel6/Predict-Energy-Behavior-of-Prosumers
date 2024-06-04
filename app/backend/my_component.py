import streamlit.components.v1 as components

# Define the path to the frontend HTML file
_component_func = components.declare_component(
    "my_component",
    path="frontend"
)

# Create a function to call the component
def my_component():
    component_value = _component_func()
    return component_value
