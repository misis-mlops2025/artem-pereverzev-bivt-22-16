import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
project = data.get('project', {})
print(f'Name: {project.get("name", "N/A")}')
print(f'Version: {project.get("version", "N/A")}')
print(f'Description: {project.get("description", "N/A")}')