import re

from html import unescape
from typing import Optional
from app.core.document_processing.html_manual_parser import HtmlManualParser
from app.core.knowledge_source.entities import HtmlManualChunk, HtmlManualDocument, HtmlManualEvent


class HtmlManualConverter:
    """
    Este servicio encapsula la conversión de manuales HTML a bloques listos para ingesta.
    """

    def __init__(
        self,
        max_html_bytes: int,
        max_image_bytes: int,
        max_chunk_chars: int,
        max_image_context_chars: int,
    ) -> None:
        self.max_html_bytes = max_html_bytes
        self.max_image_bytes = max_image_bytes
        self.max_chunk_chars = max_chunk_chars
        self.max_image_context_chars = max_image_context_chars


    def convert(self, filename: str, content: bytes, content_type: str | None) -> HtmlManualDocument:
        cleaned_filename = self._clean_filename(filename)
        self._validate_html_file(cleaned_filename, content, content_type)
        html = self._decode_html(content)
        title, events = self._parse_html(cleaned_filename, html)
        chunks = self._build_chunks(title, events)
        if not chunks:
            raise ValueError("No se ha encontrado contenido válido en el manual HTML")

        return HtmlManualDocument(
            filename=cleaned_filename,
            title=title,
            chunks=chunks,
        )


    def _clean_filename(self, filename: str) -> str:
        cleaned = filename.replace("\\", "/").split("/")[-1].strip()
        return cleaned or "manual.html"


    def _validate_html_file(self, filename: str, content: bytes, content_type: str | None) -> None:
        if not content:
            raise ValueError("El fichero HTML está vacío")
        if len(content) > self.max_html_bytes:
            raise ValueError("El fichero HTML supera el tamaño máximo permitido")

        allowed_content_types = {"text/html", "application/xhtml+xml", "application/octet-stream"}
        is_html_filename = filename.lower().endswith((".html", ".htm"))
        is_html_content_type = content_type in allowed_content_types if content_type else False
        if not is_html_filename and not is_html_content_type:
            raise ValueError("El fichero debe ser HTML")


    def _decode_html(self, content: bytes) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")


    def _parse_html(self, filename: str, html: str) -> tuple[str, list[HtmlManualEvent]]:
        parser = HtmlManualParser(max_image_bytes=self.max_image_bytes)
        parser.feed(html)
        parser.close()

        title = next(
            (
                event.text
                for event in parser.events
                if event.event_type == "heading" and event.text
            ),
            filename,
        )
        return title, parser.events


    def _build_chunks(self, title: str, events: list[HtmlManualEvent]) -> list[HtmlManualChunk]:
        raw_text = "\n\n".join(
            event.text
            for event in events
            if event.text and event.event_type in {"heading", "text"}
        )
        chunks: list[HtmlManualChunk] = []
        for content in self._split_text(raw_text):
            chunks.append(
                HtmlManualChunk(
                    index=len(chunks) + 1,
                    chunk_type="text",
                    title=title,
                    content=content,
                )
            )

        for index, event in enumerate(events):
            if not event.image:
                continue
            content = self._build_image_context(events, index, title)
            chunks.append(
                HtmlManualChunk(
                    index=len(chunks) + 1,
                    chunk_type="image",
                    title=title,
                    content=content,
                    image=event.image,
                )
            )

        return chunks


    def _split_text(self, text: str) -> list[str]:
        cleaned = self._clean_text(text)
        if not cleaned:
            return []

        chunks: list[str] = []
        current = ""
        paragraphs = re.split(r"\n\s*\n", cleaned)
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) > self.max_chunk_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                for index in range(0, len(paragraph), self.max_chunk_chars):
                    chunks.append(paragraph[index:index + self.max_chunk_chars].strip())
                continue
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) > self.max_chunk_chars:
                chunks.append(current.strip())
                current = paragraph
            else:
                current = candidate

        if current:
            chunks.append(current.strip())
        return chunks


    def _build_image_context(self, events: list[HtmlManualEvent], image_index: int, title: str) -> str:
        lines = [title]
        start_index = max(0, image_index - 6)
        end_index = min(len(events), image_index + 4)
        for event in events[start_index:end_index]:
            if event.text:
                lines.append(event.text)

        image = events[image_index].image
        if image:
            if image.alt:
                lines.append(f"Imagen: {image.alt}")
            if image.source:
                lines.append(f"Imagen referenciada: {image.source}")
            if image.skipped_reason:
                lines.append(f"Nota de imagen: {image.skipped_reason}")

        content = self._clean_text("\n".join(lines))
        if len(content) > self.max_image_context_chars:
            return content[:self.max_image_context_chars].strip()
        return content


    def _clean_text(self, text: Optional[str]) -> str:
        if not text:
            return ""

        cleaned = text

        # Eliminar contenido no visible.
        cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", cleaned)

        # Convertir saltos y bloques HTML habituales en saltos de línea.
        cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
        cleaned = re.sub(r"(?i)</?(p|div|li|tr|table|tbody|thead|section|article|h[1-6])[^>]*>", "\n", cleaned)

        # Eliminar el resto de etiquetas HTML.
        cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)

        # Decodificar entidades HTML: &nbsp;, &amp;, etc.
        cleaned = unescape(cleaned)
        cleaned = cleaned.replace("\xa0", " ")

        # Normalizar espacios y saltos de línea.
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        cleaned = cleaned.strip()

        # Limpieza final.
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

        return cleaned.strip()
