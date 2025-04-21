FROM python:3.10.16-slim-bullseye
WORKDIR /industrial_captioning_app
COPY tests ./tests
COPY requirements.txt .
ENV user=torchuser
USER root 
RUN apt-get update && apt-get install -y \
&& apt-get install -y git && pip install --no-cache-dir -r requirements.txt
RUN useradd ${user} && chown -R  ${user}:${user} . && \
mkdir -p /home/${user} && chown -R ${user}:${user} /home/${user} \
&& chmod -R 775 /home/${user}
RUN apt-get update && apt-get install -y libgl1 \
&& apt-get install -y libglib2.0-0
# download stopwords
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data'); nltk.download('punkt_tab',download_dir='/usr/local/nltk_data')"
# copy the custom stopwords
COPY custom_english /usr/local/nltk_data/corpora/stopwords/
# for updating the script fast
COPY src ./src
RUN chown -R ${user}:${user} /industrial_captioning_app
USER ${user}
WORKDIR ./src/training_and_evaluation
RUN chmod 777 .
RUN pwd && ls -a 
ENTRYPOINT ["python","./inference_demo.py"]


