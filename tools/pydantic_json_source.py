from pathlib import Path
from typing import Any, Iterable, Type

import commentjson  # type: ignore[import]
from pydantic.fields import Field, FieldInfo
from pydantic_settings import PydanticBaseSettingsSource, SettingsConfigDict


class JsonSettingsConfigDict(SettingsConfigDict):
    json_file: str | Path | Iterable[str | Path] | None
    json_file_encoding: str | None


class JsonConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source class that loads variables from a JSON file
    at the project's root.

    The keys 'json_file' and 'json_file_encoding' are used to configure the source
    """

    config: JsonSettingsConfigDict = Field(...)

    @staticmethod
    def parse_json_file(path: Path, encoding: str) -> dict[str, Any]:
        if not path.is_file():
            return {}

        with open(path, encoding=encoding) as json_file:
            return commentjson.load(json_file)

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        file_path = self.config.get("json_file")
        if file_path is None:
            return None, "", False

        here = Path(__file__).parent

        encoding = self.config.get("json_file_encoding")
        if encoding is None:
            encoding = "utf-8"
        encoding = str(encoding)

        file_content_json: dict[str, Any] = {}

        match file_path:
            case str():
                file_path = here / Path(file_path)
                file_content_json = self.parse_json_file(file_path, encoding)

            case Path():
                file_content_json = self.parse_json_file(here / file_path, encoding)

            case _:
                for path in file_path:
                    path = here / Path(path)
                    file_content_json = {
                        **file_content_json,
                        **self.parse_json_file(path, encoding),
                    }

        if len(file_content_json) == 0:
            return None, "", False

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
