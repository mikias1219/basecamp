import streamlit.components.v1 as components
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# Define the path to the JavaScript component
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
logging.debug(f"Looking for frontend assets at: {frontend_path}")
if not os.path.exists(frontend_path):
    logging.error(f"Frontend directory not found at: {frontend_path}")
    raise FileNotFoundError(f"Frontend directory not found at: {frontend_path}")

_component_func = components.declare_component(
    "mention_autocomplete",
    path=frontend_path
)

def mention_autocomplete(people: list, key: str = None):
    logging.debug(f"Calling mention_autocomplete with people: {[p['name'] for p in people]}")
    people_names = [p["name"] for p in people if p.get("name")]
    component_value = _component_func(people=people_names, key=key, default="")
    logging.debug(f"Component returned: {component_value}")
    return component_value