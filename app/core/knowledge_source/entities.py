from datetime import datetime
from dataclasses import dataclass
from typing import Any, ClassVar, Literal
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class RetrievalDocument(BaseModel):
    id: str
    score: float
    content: str
    metadata: dict[str, Any]


class RetrievedContext(BaseModel):
    query: str
    documents: list[RetrievalDocument]


@dataclass(frozen=True)
class PayloadKeys:
    metadata_key: str
    semantic_content_key: str
    lexical_content_key: str | None = None


@dataclass(frozen=True)
class KnowledgeSourceDefinition:
    id: str
    name: str
    description: str
    collection_name: str
    payload_keys: PayloadKeys
    dense_vector_name: str | None
    sparse_vector_name: str | None
    retrieval_type: Literal["semantic", "lexical", "hybrid"]


@dataclass(frozen=True)
class TicketArticleRow(BaseModel):
    ticket_id: int
    ticket_group_id: int
    ticket_priority_id: int
    ticket_state_id: int
    ticket_organization_id: int | None = None
    ticket_number: str
    ticket_title: str
    ticket_created_at: datetime
    ticket_closed_at: datetime
    ticket_customer_firstname: str | None = None
    ticket_customer_lastname: str | None = None
    ticket_customer_department: str | None = None
    ticket_customer_email: str | None = None
    ticket_creator_firstname: str | None = None
    ticket_creator_lastname: str | None = None
    ticket_creator_department: str | None = None
    ticket_creator_email: str | None = None
    article_id: int
    article_from: str | None = None
    article_to: str | None = None
    article_subject: str | None = None
    article_content_type: str | None = None
    article_body: str | None = None
    article_internal: bool
    article_created_at: datetime
    article_creator_firstname: str | None = None
    article_creator_lastname: str | None = None
    article_creator_department: str | None = None
    article_creator_email: str | None = None


@dataclass(frozen=True)
class TicketArticle(BaseModel):
    id: int
    from_email: str | None = None
    to_email: str | None = None
    subject: str | None = None
    content_type: str | None = None
    body: str | None = None
    internal: bool
    created_at: datetime
    creator_firstname: str | None = None
    creator_lastname: str | None = None
    creator_department: str | None = None
    creator_email: str | None = None


@dataclass(frozen=True)
class Ticket(BaseModel):
    id: int
    group_id: int
    priority_id: int
    state_id: int
    organization_id: int | None = None
    number: str
    title: str
    created_at: datetime
    closed_at: datetime
    customer_firstname: str | None = None
    customer_lastname: str | None = None
    customer_department: str | None = None
    customer_email: str | None = None
    creator_firstname: str | None = None
    creator_lastname: str | None = None
    creator_department: str | None = None
    creator_email: str | None = None
    articles: list[TicketArticle]


class Device(BaseModel):
    UNKNOWN_VALUES: ClassVar[set[str]] = {"desconocido", "desconocida", ""}

    id: int 
    name: str | None = Field(default=None) 
    type: str | None = Field(default=None, validation_alias=AliasChoices("device_type_id", "device_type", "type"))
    operating_system: str | None = Field(default=None, validation_alias=AliasChoices("operating_system_id", "operating_system"))
    operating_system_version: str | None = Field(default=None) 
    operating_system_serial_number: str | None = Field(default=None) 
    architecture: str | None = Field(default=None) 
    hostname: str | None = Field(default=None) 
    last_seen_ip: str | None = Field(default=None) 
    serial_number: str | None = Field(default=None) 
    manufacturer: str | None = Field(default=None) 
    model: str | None = Field(default=None) 
    description: str | None = Field(default=None) 
    comments: str | None = Field(default=None) 
    location: str | None = Field(default=None, validation_alias=AliasChoices("location_id", "location"))
    printer_model: str | None = Field(default=None, validation_alias=AliasChoices("printer_model_id", "printer_model"))
    employee: str | None = Field(default=None, validation_alias=AliasChoices("employee_id", "employee"))
    ram: float | None = Field(default=None) 
    hdd_size: float | None = Field(default=None) 
    user_logged: str | None = Field(default=None) 
    uptime: str | None = Field(default=None) 
    date_creation: str | None = Field(default=None) 

    ip_addresses: list[str] = Field(default_factory=list)
    mac_addresses: list[str] = Field(default_factory=list)
    cpus: list[str] = Field(default_factory=list)
    vlans: list[str] = Field(default_factory=list)

    @classmethod
    def _none_if_unknown(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().casefold() in cls.UNKNOWN_VALUES:
            return None
        return value

    @staticmethod
    def _collect_numbered_fields(data: dict[str, Any], field_prefix: str, amount: int) -> list[str]:
        return [
            value
            for index in range(1, amount + 1)
            if (value := data.get(f"{field_prefix}_{index}")) not in (None, "")
        ]

    @model_validator(mode="before")
    @classmethod
    def normalize_numbered_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = {
            key: cls._none_if_unknown(value)
            for key, value in data.items()
        }

        data = data.copy()
        data.setdefault("ip_addresses", cls._collect_numbered_fields(data, "ip", 4))
        data.setdefault("mac_addresses", cls._collect_numbered_fields(data, "mac", 4))
        data.setdefault("cpus", cls._collect_numbered_fields(data, "processor", 2))
        data.setdefault("vlans", cls._collect_numbered_fields(data, "vlan_id", 4))
        return data

    @field_validator("ip_addresses", "mac_addresses", "cpus", "vlans", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> Any:
        if value is None:
            return []

        if not isinstance(value, list):
            return value

        return [
            item
            for item in (cls._none_if_unknown(item) for item in value)
            if item not in (None, "")
        ]
    

@dataclass(frozen=True)
class Phone(BaseModel):
    number: str | None = None
    extension: str | None = None


@dataclass(frozen=True)
class Employee(BaseModel):
    id: int
    first_name: str | None = None
    last_name: str | None = None
    full_name: str | None = None
    normalized_full_name: str | None = None
    alias: str | None = None
    normalized_alias: str | None = None
    department: str | None = None
    emails: list[str] = Field(default_factory=list)
    phones: list[Phone] = Field(default_factory=list)


@dataclass(frozen=True)
class HtmlManualImage:
    id: str
    mime_type: str | None
    size_bytes: int
    data_url: str | None = None
    alt: str | None = None
    source: str | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True)
class HtmlManualEvent:
    event_type: str
    text: str | None = None
    image: HtmlManualImage | None = None


@dataclass(frozen=True)
class HtmlManualChunk:
    index: int
    chunk_type: str
    title: str
    content: str
    image: HtmlManualImage | None = None


@dataclass(frozen=True)
class HtmlManualDocument:
    filename: str
    title: str
    chunks: list[HtmlManualChunk]
