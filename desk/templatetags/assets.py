from __future__ import annotations

import os

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()


@register.simple_tag
def asset(path: str) -> str:
    """Static URL stamped with the file's mtime so browsers refetch edited assets."""
    url = static(path)
    located = finders.find(path)
    if not located:
        return url
    try:
        stamp = int(os.path.getmtime(located))
    except OSError:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={stamp}"
