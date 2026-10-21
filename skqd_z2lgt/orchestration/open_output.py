"""Set up the output directory and write the parameters JSON."""
import os
import logging
from typing import Optional
import yaml
from pathlib import Path
from pydantic import BaseModel
from skqd_z2lgt.orchestration.parameters import Parameters


def compare_models(model1: BaseModel, model2: BaseModel, _current: str = ''):
    for field in type(model1).model_fields:
        value1 = getattr(model1, field)
        value2 = getattr(model2, field)
        if isinstance(value1, BaseModel):
            if (mismatch := compare_models(value1, value2, f'{_current}{field}.')):
                return mismatch
        elif value1 != value2:
            return f'{_current}{field}', value1, value2


def open_output(
    parameters: Parameters | str | Path,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None
) -> Parameters:
    """Open a new output HDF5 file and set it up for the workflow, or validate an existing file.

    Args:
        parameters: Workflow parameters or pkgpath.
        overwrite: Overwrite existing parameters.json with the provided parameters.

    Returns:
        Orchestration parameters.
    """
    logger = logger or logging.getLogger(__name__)

    pkgpath = Path(parameters.pkgpath) if isinstance(parameters, Parameters) else Path(parameters)

    if overwrite or not os.path.exists(pkgpath / 'parameters.json'):
        if isinstance(parameters, str):
            raise ValueError(f'Need parameters loaded from yaml to save into {pkgpath}')
        logger.info('Creating a new file %s', pkgpath)
        try:
            os.makedirs(pkgpath)
        except FileExistsError:
            pass
        with open(pkgpath / 'parameters.json', 'w', encoding='utf-8') as out:
            out.write(parameters.model_dump_json())

    logger.info('Validating configurations in existing file %s', pkgpath)
    with open(pkgpath / 'parameters.json', 'r', encoding='utf-8') as source:
        params = Parameters.model_validate_json(source.read())
    params.pkgpath = str(pkgpath)

    if isinstance(parameters, Parameters) and (mismatch := compare_models(params, parameters)):
        raise RuntimeError(f'Saved parameters do not match at {mismatch[0]}:'
                           f' {mismatch[1]} != {mismatch[2]}')

    return params


def open_from_yaml(
    path: str | Path,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None
) -> Parameters:
    """Load the parameters from a YAML file and open the output directory at pkgpath."""
    with open(path, 'r', encoding='utf-8') as src:
        parameters = Parameters(**yaml.load(src, yaml.Loader))

    return open_output(parameters, overwrite=overwrite, logger=logger)
