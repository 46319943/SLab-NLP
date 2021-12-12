:: 打包
python setup.py sdist bdist_wheel
:: 发布到test-pypi
python -m twine upload --repository testpypi dist/*
:: 从pypi安装
pip install --index-url https://test.pypi.org/simple/ --no-deps --upgrade slab-nlp