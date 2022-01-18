FROM cv1s10n/dl-docker:cpu

COPY . /

# ENV THEANO_FLAGS=device=gpu7
RUN pip --no-cache-dir install optunity

CMD [ "python", "-u", "/hyperparam_search.py", \
"/shared/logs", \
"/shared/data/liver/LiverInputOverallSurvSimpleAll.hd5", \
"/box_constraints.0.json", \
"50", \
"--update_fn", "adam", \
"--num_epochs", "500", \
"--num_fold", "3" ]
