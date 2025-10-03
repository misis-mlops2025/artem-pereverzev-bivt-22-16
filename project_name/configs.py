from pydantic import BaseModel


class MyConfig(BaseModel):
    num_epoch: int = 1