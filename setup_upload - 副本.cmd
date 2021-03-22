python setup.py sdist bdist_wheel
python -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --no-deps --upgrade slab-nlp