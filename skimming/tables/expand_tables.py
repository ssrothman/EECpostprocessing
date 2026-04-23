from typing import Sequence


_ALL_KINEMATICS_TABLES = [
    "AK4JetKinematicsTable",
    "EventKinematicsTable",
    "ConstituentKinematicsTable",
    "CutflowTable",
    "SimonJetKinematicsTable",
]
_ALL_RES4TEE_TABLES = [
    "EECres4Obs:True,tee,total",
    "EECres4Obs:True,tee,unmatched",
    "EECres4Obs:True,tee,untransfered",
    "EECres4Obs:False,tee,total",
    "EECres4Obs:False,tee,unmatched",
    "EECres4Obs:False,tee,untransfered",
    "EECres4Transfer:tee"
]
_ALL_RES4DIPOLE_TABLES = [
    "EECres4Obs:True,dipole,total",
    "EECres4Obs:True,dipole,unmatched",
    "EECres4Obs:True,dipole,untransfered",
    "EECres4Obs:False,dipole,total",
    "EECres4Obs:False,dipole,unmatched",
    "EECres4Obs:False,dipole,untransfered",
    "EECres4Transfer:dipole"
]
_ALL_RES4TRIANGLE_TABLES = [
    "EECres4Obs:True,triangle,total",
    "EECres4Obs:True,triangle,unmatched",
    "EECres4Obs:True,triangle,untransfered",
    "EECres4Obs:False,triangle,total",
    "EECres4Obs:False,triangle,unmatched",
    "EECres4Obs:False,triangle,untransfered",
    "EECres4Transfer:triangle"
]

_ALL_RES4TEE_RECO_TABLES = [
    "EECres4Obs:False,tee,total",
]

_ALL_RES4DIPOLE_RECO_TABLES = [
    "EECres4Obs:False,dipole,total",
]

_ALL_RES4TRIANGLE_RECO_TABLES = [
    "EECres4Obs:False,triangle,total",
]

_ALL_RES4RECO_TABLES = [
    "EECres4Obs:False,tee,total",
    "EECres4Obs:False,dipole,total",
    "EECres4Obs:False,triangle,total",

]

_ALL_RES4_TABLES = _ALL_RES4TEE_TABLES + _ALL_RES4DIPOLE_TABLES + _ALL_RES4TRIANGLE_TABLES

def expand_one_table(tab : str) -> Sequence[str]:
    if tab == 'allKinematics':
        return _ALL_KINEMATICS_TABLES
    elif tab == 'allRes4':
        return _ALL_RES4_TABLES
    elif tab == 'allRes4tee':
        return _ALL_RES4TEE_TABLES
    elif tab == 'allRes4dipole':
        return _ALL_RES4DIPOLE_TABLES
    elif tab == 'allRes4triangle':
        return _ALL_RES4TRIANGLE_TABLES
    elif tab == 'allRes4reco':
        return _ALL_RES4RECO_TABLES
    else:
        return [tab]

def expand_tables(tables : Sequence[str]) -> Sequence[str]:
    result = []
    for tab in tables:
        result.extend(expand_one_table(tab))
    return result


from skimming.tables.driver import construct_table_from_string

def one_table_name(tab : str) -> str:
    try:
        return construct_table_from_string(tab).name
    except ValueError:
        return tab

def table_names(tables : Sequence[str]) -> Sequence[str]:
    return [one_table_name(tab) for tab in tables]