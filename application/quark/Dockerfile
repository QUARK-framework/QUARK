FROM python:3.9

COPY . .
RUN pip install -r .settings/requirements_full.txt

ENTRYPOINT ["python", "src/main.py"]
