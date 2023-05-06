from dataclasses import dataclass
from typing import Hashable

from pydantic import BaseModel


class PingRequest(BaseModel):
    pass


class PingResponse(BaseModel):
    pass


class LoadRequest(BaseModel):
    load: bool
    flush: bool


class LoadResponse(BaseModel):
    success: bool


@dataclass
class LoadEntry:
    uid: Hashable
    load: bool
    flush: bool
