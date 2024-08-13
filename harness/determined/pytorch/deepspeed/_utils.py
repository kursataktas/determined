import json
from typing import Any, Dict, Union, cast

from deepspeed.runtime import config_utils

from determined import util


def overwrite_deepspeed_config(
    base_ds_config: Union[str, Dict], source_ds_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Overwrite a base_ds_config with values from a source_ds_dict.

    You can use source_ds_dict to overwrite leaf nodes of the base_ds_config.
    More precisely, we will iterate depth first into source_ds_dict and if a node corresponds to
    a leaf node of base_ds_config, we copy the node value over to base_ds_config.

    Arguments:
        base_ds_config (str or Dict): either a path to a DeepSpeed config file or a dictionary.
        source_ds_dict (Dict): dictionary with fields that we want to copy to base_ds_config
    Returns:
        The resulting dictionary when base_ds_config is overwritten with source_ds_dict.
    """
    if isinstance(base_ds_config, str):
        base_ds_config = json.load(
            open(base_ds_config, "r"),
            object_pairs_hook=config_utils.dict_raise_error_on_duplicate_keys,
        )
    else:
        if not isinstance(base_ds_config, dict):
            raise TypeError("Expected string or dict for base_ds_config argument.")

    return util.merge_dicts(cast(Dict[str, Any], base_ds_config), source_ds_dict)
