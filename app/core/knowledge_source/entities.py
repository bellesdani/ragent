from datetime import datetime
from pydantic import BaseModel
from typing import Any, Literal
from dataclasses import dataclass


class RetrievalDocument(BaseModel):
    id: str
    score: float
    content: str
    metadata: dict[str, Any]


class RetrievedContext(BaseModel):
    query: str
    documents: list[RetrievalDocument]


@dataclass(frozen=True)
class KnowledgeSourceDefinition:
    id: str
    name: str
    description: str
    collection_name: str
    dense_vector_name: str | None
    sparse_vector_name: str | None
    retrieval_type: Literal["semantic", "hybrid"]



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
    name: str
    type: str
    os: str | None = None
    os_version: str | None = None
    os_serial: str | None = None
    architecture: str | None = None
    hostname: str | None = None
    current_ip: str | None = None
    ips: list[str] 
    mac_addresses: list[str] 
    vlans: list[str] 
    serial_number: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    description: str | None = None
    comments: str | None = None
    location: str | None = None
    printer_model: str | None = None
    owner: str | None = None
    ram_gb: int = 0
    disk_gb: int = 0
    cpu: list[str] 
    user: str | None = None
    last_reboot: datetime | None = None
    created_at: datetime