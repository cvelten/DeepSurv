#!/bin/bash
EXPERIMENT=liver_os_sbrtORrfa docker-compose up --build > log_liver_os_sbrtORrfa
EXPERIMENT=liver_os_sbrtVSnone docker-compose up --build > log_liver_os_sbrtVSnone
EXPERIMENT=liver_os_sbrtVSrfa docker-compose up --build > log_liver_os_sbrtVSrfa
EXPERIMENT=liver_os_rfaVSnone docker-compose up --build > log_liver_os_rfaVSnone
#
EXPERIMENT=liver_pfs_sbrtORrfa docker-compose up --build > log_liver_pfs_sbrtORrfa
EXPERIMENT=liver_pfs_sbrtVSnone docker-compose up --build > log_liver_pfs_sbrtVSnone
EXPERIMENT=liver_pfs_sbrtVSrfa docker-compose up --build > log_liver_pfs_sbrtVSrfa
EXPERIMENT=liver_pfs_rfaVSnone docker-compose up --build > log_liver_pfs_rfaVSnone
