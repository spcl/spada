import enum
import json
import dataclasses
from typing import Type, TypeVar, Any, Dict, Union, get_origin, get_args
import sys

T = TypeVar('T')


class DataclassEncoder(json.JSONEncoder):

    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            result = self._convert_dataclass(obj)
            return result
        if isinstance(obj, enum.Enum):
            return obj.name
        return super().default(obj)

    def _convert_dataclass(self, obj):
        """Convert a dataclass to dict, recursively handling nested dataclasses."""
        result = {}
        result['__dataclass_type__'] = obj.__class__.__name__

        for field in dataclasses.fields(obj):
            field_value = getattr(obj, field.name)
            result[field.name] = self._convert_value(field_value)

        return result

    def _convert_value(self, value):
        """Convert a value, handling dataclasses, enums, lists recursively."""
        if dataclasses.is_dataclass(value):
            return self._convert_dataclass(value)
        elif isinstance(value, enum.Enum):
            return value.name
        elif isinstance(value, list):
            return [self._convert_value(item) for item in value]
        elif isinstance(value, dict):
            # Convert dictionary keys and values recursively
            # JSON keys must be strings, so convert enum keys to their names
            return {(k.name if isinstance(k, enum.Enum) else k): self._convert_value(v) for k, v in value.items()}
        else:
            return value


def dataclass_decoder(dataclass_type: Type[T]) -> callable:

    def decode_dataclass(obj: Dict[str, Any]) -> T:
        # Make a copy to avoid modifying the original
        obj_copy = obj.copy()

        # Remove the type marker as it's not needed for instantiation
        obj_copy.pop('__dataclass_type__', None)
        # Handle nested dataclasses if needed
        for field in dataclasses.fields(dataclass_type):
            field_name = field.name
            if field_name in obj_copy:
                # Handle Union types that might contain dataclasses
                if _is_union_type(field.type):
                    obj_copy[field_name] = _decode_union_field(obj_copy[field_name], field.type)
                # Handle direct dataclass fields
                elif dataclasses.is_dataclass(field.type):
                    nested_decoder = dataclass_decoder(field.type)
                    obj_copy[field_name] = nested_decoder(obj_copy[field_name])
                # Handle lists that might contain dataclasses or unions
                elif _is_list_type(field.type):
                    obj_copy[field_name] = _decode_list_field(obj_copy[field_name], field.type)
                # Handle dicts that might contain dataclasses or unions
                elif _is_dict_type(field.type):
                    obj_copy[field_name] = _decode_dict_field(obj_copy[field_name], field.type)
                # Convert enum fields if necessary
                elif isinstance(field.type, type) and issubclass(field.type, enum.Enum):
                    if isinstance(obj_copy[field_name], str):
                        obj_copy[field_name] = field.type[obj_copy[field_name]]
        return dataclass_type(**obj_copy)

    return decode_dataclass


def _is_union_type(field_type) -> bool:
    """Check if a field type is a Union type."""
    return get_origin(field_type) is Union


def _is_list_type(field_type) -> bool:
    """Check if a field type is a List type."""
    origin = get_origin(field_type)
    return origin is list or (hasattr(field_type, '__name__') and field_type.__name__ == 'list')


def _is_dict_type(field_type) -> bool:
    """Check if a field type is a Dict type."""
    origin = get_origin(field_type)
    return origin is dict or (hasattr(field_type, '__name__') and field_type.__name__ == 'dict')


def _decode_union_field(value: Any, union_type: Type) -> Any:
    """Decode a field that has a Union type annotation."""
    if value is None:
        return None

    # If the value is a dictionary with __dataclass_type__, use that to determine the type
    if isinstance(value, dict) and '__dataclass_type__' in value:
        dataclass_type_name = value['__dataclass_type__']

        # Find the matching type from the Union args
        union_args = get_args(union_type)
        for arg_type in union_args:
            if (dataclasses.is_dataclass(arg_type) and arg_type.__name__ == dataclass_type_name):
                decoder = dataclass_decoder(arg_type)
                return decoder(value)

    # For non-dataclass Union members (like str, int, etc.), return as-is
    return value


def _decode_list_field(value: list, list_type: Type) -> list:
    """Decode a field that has a List type annotation."""
    if not value:
        return value

    # Get the element type of the list
    args = get_args(list_type)
    if not args:
        return value

    element_type = args[0]
    result = []

    for item in value:
        if _is_union_type(element_type):
            result.append(_decode_union_field(item, element_type))
        elif dataclasses.is_dataclass(element_type):
            decoder = dataclass_decoder(element_type)
            result.append(decoder(item))
        elif isinstance(element_type, type) and issubclass(element_type, enum.Enum):
            result.append(element_type[item])
        else:
            result.append(item)

    return result


def _decode_dict_field(value: dict, dict_type: Type) -> dict:
    """Decode a field that has a Dict type annotation."""
    if not value:
        return value

    # Get the key and value types of the dict
    args = get_args(dict_type)
    if not args or len(args) < 2:
        return value

    key_type, value_type = args[0], args[1]
    result = {}

    for k, v in value.items():
        # Decode the key if necessary (usually keys are strings, but could be enums)
        decoded_key = k
        if isinstance(key_type, type) and issubclass(key_type, enum.Enum):
            decoded_key = key_type[k]

        # Decode the value based on its type
        if _is_union_type(value_type):
            decoded_value = _decode_union_field(v, value_type)
        elif dataclasses.is_dataclass(value_type):
            decoder = dataclass_decoder(value_type)
            decoded_value = decoder(v)
        elif isinstance(value_type, type) and issubclass(value_type, enum.Enum):
            decoded_value = value_type[v]
        else:
            decoded_value = v

        result[decoded_key] = decoded_value

    return result


# Save to JSON
def save_to_json(dataclass_instance, filepath):
    """
    Save a dataclass instance to a JSON file.

    :param dataclass_instance: The dataclass instance to save.
    :param filepath: The path to the JSON file where the dataclass will be saved.
    """
    with open(filepath, 'w') as f:
        json.dump(dataclass_instance, f, cls=DataclassEncoder, indent=2)


# Load from JSON
def load_from_json(dataclass_type: Type[T], filepath) -> T:
    """
    Load a dataclass instance from a JSON file.

    :param dataclass_type: The type of the dataclass to load.
    :param filepath: The path to the JSON file from which the dataclass will be loaded.
    :return: An instance of the dataclass.
    """
    with open(filepath, 'r') as f:
        json_data = json.load(f)
    return dataclass_decoder(dataclass_type)(json_data)
