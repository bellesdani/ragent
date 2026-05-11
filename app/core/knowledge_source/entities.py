from datetime import datetime
from typing import Any, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field


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


class TicketArticle(BaseModel):
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


class Ticket(BaseModel):
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

    articles: list[TicketArticle]


class Device(BaseModel):
    id: int
    name: str | None = None
    type: str | None = None
    os: str | None = None
    os_version: str | None = None
    os_serial: str | None = None
    architecture: str | None = None
    hostname: str | None = None
    current_ip: str | None = None
    ips: list[str] = Field(default_factory=list)
    mac_addresses: list[str] = Field(default_factory=list)
    vlans: list[str] = Field(default_factory=list)
    serial_number: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    description: str | None = None
    comments: str | None = None
    location: str | None = None
    printer_model: str | None = None
    owner: str | None = None
    ram_gb: float | None = None
    disk_gb: float | None = None
    cpu: list[str] = Field(default_factory=list)
    user: str | None = None
    last_reboot: str | None = None
    created_at: str | None = None


class Phone(BaseModel):
    number: str | None = None
    extension: str | None = None


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
