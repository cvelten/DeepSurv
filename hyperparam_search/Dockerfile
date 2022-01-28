FROM cv1s10n/dl-docker:cpu

COPY . /

RUN pip --no-cache-dir install optunity

CMD [ "python", "-u", "/hyperparam_search.py", \
"/shared/logs", \
# "/shared/data/liver_os_sbrt/220123Liver-OS_TxSBRT.hd5", \ # 1
# "/shared/data/liver_os_rfa/220123Liver-OS_TxRFA.hd5", \
# "/shared/data/liver_os_sbrtORrfa/220123Liver-OS_TxSBRT-OR-RFA.hd5", \
# "/shared/data/liver_pfs_sbrt/220123Liver-PFS_TxSBRT.hd5", \
# "/shared/data/liver_pfs_rfa/220123Liver-PFS_TxRFA.hd5", \
"/shared/data/liver_pfs_sbrtORrfa/220123Liver-PFS_TxSBRT-OR-RFA.hd5", \
# "/shared/data/liver/LiverInputOverallSurvSimpleAll.hd5", \
"/box_constraints.0.json", \
"50", \
"--update_fn", "adam", \
"--num_epochs", "500", \
"--num_fold", "3" ]
