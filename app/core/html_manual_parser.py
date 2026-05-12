import re
import base64
import hashlib
import binascii

from html import unescape
from html.parser import HTMLParser
from app.core.knowledge_source.entities import HtmlManualEvent, HtmlManualImage


class HtmlManualParser(HTMLParser):
    block_tags = {
        "article", "section", "div", "p", "br", "li", "ul", "ol",
        "table", "tbody", "thead", "tr", "td", "th",
    }
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    ignored_tags = {"script", "style"}

    def __init__(self, max_image_bytes: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_image_bytes = max_image_bytes
        self.events: list[HtmlManualEvent] = []
        self._text_parts: list[str] = []
        self._text_type = "text"
        self._ignored_depth = 0


    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.ignored_tags:
            self._flush_text()
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "img":
            self._flush_text()
            image = self._build_image(attrs)
            if image:
                self.events.append(HtmlManualEvent(event_type="image", image=image))
            return
        if tag in self.heading_tags:
            self._flush_text()
            self._text_type = "heading"
        elif tag in self.block_tags:
            self._flush_text()


    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.ignored_tags and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag in self.heading_tags:
            self._flush_text()
            self._text_type = "text"
        elif tag in self.block_tags:
            self._flush_text()


    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = self._clean_inline_text(data)
        if text:
            self._text_parts.append(text)


    def close(self) -> None:
        super().close()
        self._flush_text()


    def _flush_text(self) -> None:
        if not self._text_parts:
            return
        text = self._clean_inline_text(" ".join(self._text_parts))
        self._text_parts = []
        if text:
            self.events.append(HtmlManualEvent(event_type=self._text_type, text=text))


    def _build_image(self, attrs: list[tuple[str, str | None]]) -> HtmlManualImage | None:
        attr_map = {name.lower(): value for name, value in attrs if value is not None}
        source = attr_map.get("src")
        if not source:
            return None

        alt = self._clean_inline_text(attr_map.get("alt", ""))
        if not source.lower().startswith("data:image/"):
            image_id = hashlib.sha256(source.encode("utf-8")).hexdigest()
            return HtmlManualImage(
                id=image_id,
                mime_type=None,
                size_bytes=0,
                alt=alt or None,
                source=source[:1000],
                skipped_reason="La imagen externa no se descarga durante la ingesta",
            )

        try:
            header, encoded = source.split(",", 1)
        except ValueError:
            image_id = hashlib.sha256(source.encode("utf-8")).hexdigest()
            return HtmlManualImage(
                id=image_id,
                mime_type=None,
                size_bytes=0,
                alt=alt or None,
                skipped_reason="La imagen embebida no tiene un formato data URL válido",
            )

        mime_type = header[5:].split(";")[0].lower()
        if "base64" not in header.lower():
            image_id = hashlib.sha256(source.encode("utf-8")).hexdigest()
            return HtmlManualImage(
                id=image_id,
                mime_type=mime_type,
                size_bytes=0,
                alt=alt or None,
                skipped_reason="La imagen embebida no está codificada en base64",
            )

        encoded = re.sub(r"\s+", "", encoded)
        try:
            image_bytes = base64.b64decode(encoded, validate=False)
        except (ValueError, binascii.Error):
            image_id = hashlib.sha256(source.encode("utf-8")).hexdigest()
            return HtmlManualImage(
                id=image_id,
                mime_type=mime_type,
                size_bytes=0,
                alt=alt or None,
                skipped_reason="La imagen embebida no se ha podido decodificar",
            )

        image_id = hashlib.sha256(image_bytes).hexdigest()
        if len(image_bytes) > self.max_image_bytes:
            return HtmlManualImage(
                id=image_id,
                mime_type=mime_type,
                size_bytes=len(image_bytes),
                alt=alt or None,
                skipped_reason="La imagen supera el tamaño máximo permitido",
            )

        return HtmlManualImage(
            id=image_id,
            mime_type=mime_type,
            size_bytes=len(image_bytes),
            data_url=f"data:{mime_type};base64,{encoded}",
            alt=alt or None,
        )


    def _clean_inline_text(self, text: str) -> str:
        cleaned = unescape(text)
        cleaned = cleaned.replace("\xa0", " ")
        cleaned = re.sub(r"[ \t\r\f\v\n]+", " ", cleaned)
        return cleaned.strip()
