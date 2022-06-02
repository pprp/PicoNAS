coverage run --branch --source lib -m pytest tests/
coverage xml
coverage report -m
