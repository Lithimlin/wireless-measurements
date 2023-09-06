import json
from pathlib import Path
from typing import Any, Type

from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource


class JsonConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source class that loads variables from a JSON file
    at the project's root.

    The keys 'json_file' and 'json_file_encoding' are used to configure the source
    """

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        if self.config.get("json_file") is None:
            return None, "", False

        here = Path(__file__).parent
        path = here / Path(self.config.get("json_file"))
        encoding = self.config.get("json_file_encoding")

        if not path.is_file():
            return None, "", False

        if encoding is None:
            encoding = "utf-8"

        with open(path, encoding=encoding) as json_file:
            file_content_json = json.load(json_file)

        field_value = file_content_json.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> dict[str, Any]:
        d: dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            field_value = self.prepare_field_value(
                field_name, field, field_value, value_is_complex
            )
            if field_value is not None:
                d[field_key] = field_value

        return d
