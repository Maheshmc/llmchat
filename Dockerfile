# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install -r /app/requirements.txt


COPY ./chainlit.md /app/chainlit.md
COPY ./.chainlit /app/.chainlit
COPY ./app.py /app/app.py
COPY ./.env /app/.env

# EXPOSE 8000

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["chainlit run", "app.py"]
CMD ["chainlit", "run", "/app/app.py"]
# CMD ["bin", "bash"]
