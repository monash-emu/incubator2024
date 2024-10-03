import yaml
from IPython.display import Markdown, display
from tb_incubator.constants import project_path


def load_report_section(file_path, section):
    # Load the report sections from the YAML file
    with open(project_path / file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Retrieve the specified section
    section_text = data.get(section, "Section not found.")
    
    # Display the section text as markdown in a Jupyter notebook
    display(Markdown(section_text))

