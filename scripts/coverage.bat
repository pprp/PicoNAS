coverage run --branch --source nanonas -m pytest tests/
coverage xml
coverage report -m
