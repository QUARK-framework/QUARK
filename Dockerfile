FROM continuumio/anaconda3

RUN apt-get update && \
	apt-get install -y \
		g++ \
		gcc \
		nano \
		sudo \
		vim

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "quark", "/bin/bash", "-c"]

WORKDIR /quark
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "quark", "python", "src/main.py", "--config", "config.yml"]
